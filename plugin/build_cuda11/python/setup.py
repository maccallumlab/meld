# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved

from distutils.core import setup
from distutils.extension import Extension
import os
import platform

version = '0.5.0'

openmm_dir = '/home/liweichang/program/OpenMM_dev/bin_cuda11'
meldplugin_header_dir = '/home/liweichang/program/meld_cuda11_dev/plugin/openmmapi/include'
meldplugin_library_dir = '/home/liweichang/program/meld_cuda11_dev/plugin/build_cuda11'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']


extension = Extension(name='_meldplugin',
                      version=version,
                      sources=['MeldPluginWrapper.cpp'],
                      libraries=['OpenMM', 'MeldPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), meldplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), meldplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='meldplugin',
      version=version,
      py_modules=['meldplugin'],
      ext_modules=[extension],
     )
