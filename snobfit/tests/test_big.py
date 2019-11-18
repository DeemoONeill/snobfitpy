""" Unit tests for Snobfit for a big dimension
Author: Ziwen Fu
Nov 2008
"""

import unittest
import numpy
try:
    from snobfit.snobfit import snobfit
except:
    from snobfit import snobfit


def big20(x):
    f = 0.0 
    for i in xrange(20):
        f += (x[i]-i*0.5)**2

    return f


# ---------------------------------------------------------------------
class test_Snobfit(unittest.TestCase):
    def setUp(self):
        pass

    def test_big20(self):
        # big dimension
        n = 20
        u = numpy.array( range(0,n) )*0.5 - 3
        v = numpy.array( range(0,n) )*0.5 + 2
        x0    = numpy.array( range(0,n) )*0.5 + 0.3
        fglob = 0
        xglob = numpy.array( range(0,n) ) * 0.5
        xbest, fbest, ncall = snobfit(big20,
                                      x0,
                                      (u, v),
                                      fglob=fglob,
                                      retall=1
                                      )
        assert abs(fbest-fglob) < 1.0e-6
        


#---------------------------------------------
if __name__ == '__main__':
    unittest.main()
