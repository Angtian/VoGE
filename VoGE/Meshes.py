import torch
import torch.nn as nn


class GaussianMeshesNaive:
    def __init__(self, verts, sigmas, radians=None):
        self.verts = verts
        self.sigmas = sigmas
        self.radians = radians

    def to(self, device):
        self.verts = self.verts.to(device)
        self.sigmas = self.sigmas.to(device)

        if self.radians is not None:
            self.radians = self.radians.to(device)

        return self

    def __call__(self):
        return self.verts, self.sigmas, self.radians

    def __getitem__(self, item):
        if self.radians is not None:
            return GaussianMeshesNaive(self.verts[item], self.sigmas[item], self.radians[item])
        else:
            return GaussianMeshesNaive(self.verts[item], self.sigmas[item], None)


class GaussianMeshes(nn.Module):
    def __init__(self, verts, sigmas, radians=None, gradianted_args=None):
        super(GaussianMeshes, self).__init__()
        if gradianted_args is None:
            gradianted_args = [True] * 3

        self.gradianted_args = gradianted_args
        self.verts = nn.Parameter(verts, requires_grad=gradianted_args[0])
        self.sigmas = nn.Parameter(sigmas, requires_grad=gradianted_args[1])

        if radians is not None:
            self.radians = nn.Parameter(radians, requires_grad=gradianted_args[2])
        else:
            self.radians = None
            self.gradianted_args[2] = False

    def grad_parameters(self):
        out = []
        out += [self.verts] if self.gradianted_args[0] else []
        out += [self.sigmas] if self.gradianted_args[1] else []
        out += [self.radians] if self.gradianted_args[2] else []
        return tuple(out)

    def forward(self):
        return self.verts, self.sigmas, self.radians


DeformedGaussianMeshes = GaussianMeshes
