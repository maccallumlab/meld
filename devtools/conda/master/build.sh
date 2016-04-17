#!/bin/bash

python setup.py install

cd docs
pip install msmb_theme==1.2.0 numpydoc
make html
