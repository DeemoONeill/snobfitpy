#!/usr/bin/env python

from setuptools import setup, find_packages
import fix_setuptools_chmod
import snobfit.version as ver

dist = setup(
        name        = 'snobfit',
        version     = ver.version, 
        packages    = find_packages(),
        #package_data = {'snobfit' : ['tests/*']},
        install_requires = ['pyparsing'],
        )
    

