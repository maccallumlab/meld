# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved

from distutils.core import setup
from distutils.extension import Extension
import os
import platform

version = '0.4.16'

openmm_dir = '@OPENMM_DIR@'
meldplugin_header_dir = '@MELDPLUGIN_HEADER_DIR@'
meldplugin_library_dir = '@MELDPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

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
