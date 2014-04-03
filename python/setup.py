from distutils.core import setup
from distutils.extension import Extension
import os
import sys

openmm_dir = '@OPENMM_DIR@'
exampleplugin_header_dir = '@EXAMPLEPLUGIN_HEADER_DIR@'
exampleplugin_library_dir = '@EXAMPLEPLUGIN_LIBRARY_DIR@'

extension = Extension(name='_exampleplugin',
                      sources=['ExamplePluginWrapper.cpp'],
                      libraries=['OpenMM', 'ExamplePlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), exampleplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), exampleplugin_library_dir]
                     )

setup(name='exampleplugin',
      version='1.0',
      py_modules=['exampleplugin'],
      ext_modules=[extension],
     )
