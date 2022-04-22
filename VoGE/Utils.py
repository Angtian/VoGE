import torch
import threading
from collections.abc import Iterable
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper
from typing import List, Optional


def eye_like(tensor: torch.Tensor):
    return torch.eye(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype).view((-1, ) * (len(tensor.shape) - 2) + (tensor.shape[-1],) * 2).expand(tensor.shape[0:-2] + (-1, ) * 2)


def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int=1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert len(ind.shape) > dim, "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(*tuple([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1, ] * (len(target.shape) - dim)))

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1, ) * (dim + 1), *target.shape[(dim + 1)::])

    return torch.gather(target, dim=dim, index=ind_pad)


def ind_fill(target: torch.Tensor, ind: torch.Tensor, src: [torch.Tensor, float, int], dim: int=1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :param src: value to fill
    :return: sel_target [... (k), M, ...]
    """
    assert len(ind.shape) > dim, "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(*tuple([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1, ] * (len(target.shape) - dim)))

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1, ) * (dim + 1), *target.shape[(dim + 1)::])

    if isinstance(src, torch.Tensor):
        return target.scatter(dim=dim, index=ind_pad, src=src)
    else:
        return target.scatter(dim=dim, index=ind_pad, value=src)


class Reshaper(object):
    def __init__(self, tar_shape, tar_index):
        self.tar_shape = tar_shape
        self.tar_index = tar_index

    def __call__(self, x_):
        if isinstance(x_, list) or isinstance(x_, tuple):
            if len(x_) == 0:
                return tuple()
            if isinstance(x_[0], float) or isinstance(x_[0], int):
                return sum(x_[0])
            if torch.is_tensor(x_[0]) and len(x_[0].shape) == 0:
                return torch.sum(torch.stack(x_))
            x_ = torch.cat(x_, dim=self.tar_index)

        if x_ is not None:
            return x_.view(*self.tar_shape + x_.shape[self.tar_index + 1:])
        else:
            return None


class Batchifier(object):
    def __init__(self, batch_size: int, batch_args: [str, list, tuple], target_dims: [int, list, tuple, None] = None,
                 remain_dims: [int, list, tuple, None] = None):
        """
        Automatically batchify a process. remain_dims must be placed at start and end of the shape.
        :param batch_size: batch size
        :param batch_args: name of args to batchify
        :param target_dims: the raveled dims
        :param remain_dims: the unraveled dims
        """
        if isinstance(batch_args, str):
            batch_args = (batch_args,)

        if target_dims is not None:
            if isinstance(target_dims, int):
                target_dims = (target_dims,)
            self.target_dims = tuple(target_dims)
            self.remain_dims = None
        else:
            if isinstance(remain_dims, int):
                remain_dims = (remain_dims,)
            self.remain_dims = tuple(remain_dims)
            self.target_dims = None

        self.batch_size = batch_size
        self.batch_args = tuple(batch_args)

        assert len(self.batch_args) > 0

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)

            total_len = -1

            recorded_shape = None
            save_idx = None
            for k in self.batch_args:
                get = kwargs[k]

                assert isinstance(get, torch.Tensor)

                if self.target_dims is not None:
                    this_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.target_dims])
                    other_dims = tuple([i for i in range(len(get.shape)) if i not in this_dims])
                else:
                    other_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.remain_dims])
                    this_dims = tuple([i for i in range(len(get.shape)) if i not in other_dims])

                to_shape = [get.shape[i] if i in other_dims else -1 for i in range(len(get.shape))]
                t_l = len(to_shape)

                for i in range(t_l - 1):
                    if to_shape[t_l - 1 - i] == -1 and to_shape[t_l - 2 - i] == -1:
                        del to_shape[t_l - 1 - i]

                assert to_shape.count(-1) == 1

                to_record_shape = get.shape[0:to_shape.index(-1) + len(this_dims)]
                if recorded_shape is None:
                    recorded_shape = tuple(to_record_shape)
                    save_idx = to_shape.index(-1)
                else:
                    assert recorded_shape == tuple(to_record_shape)

                kwargs[k] = get.view(*to_shape)

                total_len = kwargs[k].shape[to_shape.index(-1)]

            assert total_len >= 0, 'No batchify parameters found!'

            reshape_foo = Reshaper(tar_shape=recorded_shape, tar_index=save_idx)

            out = []
            # for i in range((total_len - 1) // self.batch_size + 1):
            for i in range((total_len - 1) // self.batch_size + 1):
                this_kwargs = dict()
                for k in kwargs.keys():
                    this_kwargs[k] = kwargs[k]
                    if k in self.batch_args:
                        exec(
                            'this_kwargs[k] = this_kwargs[k][%si * self.batch_size: (i + 1) * self.batch_size]' % ''.join(
                                (':, ',) * save_idx))

                out.append(func(*args, **this_kwargs))

            if isinstance(out[0], tuple):
                out_reshaped = [[] for _ in range(len(out[0]))]
                for this_out in out:
                    for i in range(len(out[0])):
                        out_reshaped[i].append(this_out[i])

                return tuple([reshape_foo(this_row) for this_row in out_reshaped])
            else:
                return reshape_foo(out)

        return wrapper


class DataParallelBatchifier(object):
    def __init__(self, batch_size: int, batch_args: [str, list, tuple], target_dims: [int, list, tuple, None] = None,
                 remain_dims: [int, list, tuple, None] = None, device: torch.device = None):
        """
        Automatically batchify a process. remain_dims must be placed at start and end of the shape.
        :param batch_size: batch size
        :param batch_args: name of args to batchify
        :param target_dims: the raveled dims
        :param remain_dims: the unraveled dims
        :param device: device to use, use all gpu if None
        """
        if isinstance(batch_args, str):
            batch_args = (batch_args,)

        if target_dims is not None:
            if isinstance(target_dims, int):
                target_dims = (target_dims,)
            self.target_dims = tuple(target_dims)
            self.remain_dims = None
        else:
            if isinstance(remain_dims, int):
                remain_dims = (remain_dims,)
            self.remain_dims = tuple(remain_dims)
            self.target_dims = None

        if device is None:
            self.device = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        else:
            assert isinstance(device, Iterable), 'Device must be iterable, get %s.' % str(device)
            self.device = device

        self.n_gpus = len(self.device)

        self.batch_size = batch_size
        self.batch_args = tuple(batch_args)

        assert len(self.batch_args) > 0

    def __call__(self, func):
        def _worker(lock, foo, results, device_idx, this_device, args, kwargs, autocast_enabled):
            try:
                with torch.cuda.device(this_device), autocast(enabled=autocast_enabled):
                    # this also avoids accidental slicing of `input` if it is a Tensor
                    output = foo(*args, **kwargs)
                with lock:
                    results[device_idx] = output
            except Exception:
                with lock:
                    results[device_idx] = ExceptionWrapper(
                        where="in replica {} on device {}".format(device_idx, this_device))

        def wrapper(*args, **kwargs):
            kwargs = dict(kwargs)

            total_len = -1

            recorded_shape = None
            save_idx = None
            current_device = None
            lock = threading.Lock()
            grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

            for k in self.batch_args:
                get = kwargs[k]

                assert isinstance(get, torch.Tensor)

                if current_device is None:
                    current_device = get.device
                else:
                    assert current_device == get.device, 'args must be on same device, but get %s, %s' % (
                    str(current_device), str(get.device))

                if self.target_dims is not None:
                    this_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.target_dims])
                    other_dims = tuple([i for i in range(len(get.shape)) if i not in this_dims])
                else:
                    other_dims = tuple([d if d >= 0 else len(get.shape) + d for d in self.remain_dims])
                    this_dims = tuple([i for i in range(len(get.shape)) if i not in other_dims])

                to_shape = [get.shape[i] if i in other_dims else -1 for i in range(len(get.shape))]
                t_l = len(to_shape)

                for i in range(t_l - 1):
                    if to_shape[t_l - 1 - i] == -1 and to_shape[t_l - 2 - i] == -1:
                        del to_shape[t_l - 1 - i]

                assert to_shape.count(-1) == 1

                to_record_shape = get.shape[0:to_shape.index(-1) + len(this_dims)]
                if recorded_shape is None:
                    recorded_shape = tuple(to_record_shape)
                    save_idx = to_shape.index(-1)
                else:
                    assert recorded_shape == tuple(to_record_shape)

                kwargs[k] = get.view(*to_shape)

                total_len = kwargs[k].shape[to_shape.index(-1)]

            assert total_len >= 0, 'No batchify parameters found!'

            reshape_foo = Reshaper(tar_shape=recorded_shape, tar_index=save_idx)

            out = []
            for i in range((total_len - 1) // self.batch_size + 1):
                sub_batch_size = (self.batch_size - 1) // self.n_gpus + 1 if i != total_len // self.batch_size else (total_len % self.batch_size - 1) // self.n_gpus + 1

                results = dict()
                threads = []
                for j, this_device in enumerate(self.device):
                    this_kwargs = dict()
                    for k in kwargs.keys():
                        if k in self.batch_args:
                            start_idx = i * self.batch_size + min(j * sub_batch_size, self.batch_size)
                            end_idx = i * self.batch_size + min((j + 1) * sub_batch_size, self.batch_size)
                            exec('this_kwargs[k] = kwargs[k][%s start_idx: end_idx]' % ''.join((':, ', ) * save_idx))
                        else:
                            this_kwargs[k] = kwargs[k]
                        if torch.is_tensor(this_kwargs[k]):
                            this_kwargs[k] = this_kwargs[k].to(this_device)

                    # print(this_kwargs[self.batch_args[0]].shape)

                    if this_kwargs[self.batch_args[0]].shape[save_idx] != 0:
                        threads.append(threading.Thread(target=_worker, args=(lock, func, results, j, this_device, args, this_kwargs, autocast_enabled)))

                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                for j, _ in enumerate(self.device):
                    if not j in results.keys():
                        continue
                    if isinstance(results[j], ExceptionWrapper):
                        results[j].reraise()
                    else:
                        this_out = results[j]
                        if torch.is_tensor(this_out):
                            out.append(this_out.to(current_device))
                        else:
                            out.append(tuple([t.to(current_device) for t in this_out]))

            if isinstance(out[0], tuple):
                out_reshaped = [[] for _ in range(len(out[0]))]
                for this_out in out:
                    for i in range(len(out[0])):
                        out_reshaped[i].append(this_out[i])

                return tuple([reshape_foo(this_row) for this_row in out_reshaped])
            else:
                return reshape_foo(out)

        return wrapper


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float:
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


if __name__ == '__main__':
    def foo(x, y):
        return x + 1, y - 1


    batchifier = DataParallelBatchifier(6, batch_args=('x', 'y'), remain_dims=(0, 3))

    foo_batched = batchifier(foo)
    y = torch.ones((6, 5, 2, 2))
    x = torch.arange(120).view(6, 5, 2, 2)
    get = foo_batched(x=x, y=y)
    get2 = foo(x=x, y=y)

    print(get[0].shape)