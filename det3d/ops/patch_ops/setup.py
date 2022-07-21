from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='patch_ops',
    ext_modules=[
        CUDAExtension('patch_ops_cuda', [
            'src/patch_ops_api.cpp',
            'src/patch_query.cpp',
            'src/patch_query_gpu.cu',
            'src/roipatch_dfvs_pool3d.cpp',
            'src/roipatch_dfvs_pool3d_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-I /usr/local/cuda/include'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
