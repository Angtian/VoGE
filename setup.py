import glob
import os
import runpy
import warnings
from typing import List, Optional

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

extra_compile_args = {"cxx": ["-std=c++14"]}
define_macros = []

CUDA_HOME = '/usr/local/cuda'

force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
    extension = CUDAExtension
    # sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    nvcc_args = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    if nvcc_flags_env != "":
        nvcc_args.extend(nvcc_flags_env.split(" "))

    # It's better if pytorch can do this by default ..
    CC = os.environ.get("CC", None)
    if CC is not None:
        CC_arg = "-ccbin={}".format(CC)
        if CC_arg not in nvcc_args:
            if any(arg.startswith("-ccbin") for arg in nvcc_args):
                raise ValueError("Inconsistent ccbins")
            nvcc_args.append(CC_arg)

    extra_compile_args["nvcc"] = nvcc_args
else:
    print('Cuda is not available!')

ext_modules = [
    extension('VoGE._C', [
        'VoGE/csrc/ext.cpp',
        'VoGE/csrc/rasterize_coarse/rasterize_coarse.cu',
        'VoGE/csrc/ray_trace_voge/ray_trace_voge.cu',
        'VoGE/csrc/sample_voge/sample_voge.cu'
    ],
    include_dirs=['VoGE/csrc'],
    define_macros=define_macros,
    extra_compile_args=extra_compile_args
    )
]

INSTALL_REQUIREMENTS = ['numpy', 'torch', 'plyfile', ]

setup(
    name='VoGE',
    description='Volume Gaussian Ellepsoids',
    author='AuthorVoGE',
    packages=find_packages(exclude=("demo", "data")),
    license='MIT License',
    version='0.1.1',
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)