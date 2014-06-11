from distutils.core import setup
from distutils.extension import Extension
import os
import sys

openmm_dir = '@OPENMM_DIR@'
meldplugin_header_dir = '@MELDPLUGIN_HEADER_DIR@'
meldplugin_library_dir = '@MELDPLUGIN_LIBRARY_DIR@'

extension = Extension(name='_meldplugin',
                      sources=['MeldPluginWrapper.cpp'],
                      libraries=['OpenMM', 'MeldPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), meldplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), meldplugin_library_dir]
                     )

setup(name='meldplugin',
      version='1.0',
      py_modules=['meldplugin'],
      ext_modules=[extension],
     )
