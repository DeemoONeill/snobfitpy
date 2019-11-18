#!/usr/bin/env python

import os
import sys
from os.path import join
import numpy


def configuration(parent_package='',
                  top_path=None
                  ):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration('snobfit', parent_package, top_path)
     
    #config.add_subpackage('testfun')
    #config.add_subpackage('hsfun')

    #config.make_svn_version_py()  # installs __svn_version__.py

    # Build __config__ file showing the configuration parameters for
    # the various subpackages.  As of this writing it is empty.
    #config.make_config_py()
 
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    if  len(sys.argv) == 1:
        sys.argv.append('install')
    setup(**configuration(top_path='').todict() )
