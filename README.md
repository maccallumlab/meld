# meld #

Modeling with limited data

# Prerequisites #

* [NetCDF4-python](https://code.google.com/p/netcdf4-python/) -- `pip install netCDF4`
* [mpi4py](http://mpi4py.scipy.org) -- `pip install mpi4py`
* [Openmm_Meld](https://github.com/laufercenter/OpenMM_Meld) -- follow instructions on github page

# VirtualEnv Recommended #

* `pip install virtualenv`
* `mkdir VirtualEnv`
* `cd VirtualEnv`
* `virtualenv --system-site-packages Meld`
* `source Meld/bin/activate`
* If needed: `pip install netCDF4`
* If needed: `pip install mpi4py`
* Install OpenMM_Meld.
* Change to Meld install directory
* `python setup.py install`
