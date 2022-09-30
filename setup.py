import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':

    setup(
        name='graphrcnn',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'easydict',
            'pyyaml',
            'tqdm',
            'terminaltables',
            'einops',
        ],

        author='Honghui Yang',
        author_email='yanghonghui@zju.edu.cn',
        license='MIT License',
        packages=find_packages(exclude=['tools']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='det3d.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='patch_ops_cuda',
                module='det3d.ops.patch_ops',
                sources=[
                    'src/patch_ops_api.cpp',
                    'src/patch_query.cpp',
                    'src/patch_query_gpu.cu',
                    'src/roipatch_dfvs_pool3d.cpp',
                    'src/roipatch_dfvs_pool3d_gpu.cu',
                ],
            ),
        ],
    )
