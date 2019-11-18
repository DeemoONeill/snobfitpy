""" Unit tests for Snobfit for a soft constraint

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

def hsf18(x):
    f = 0.01*x[0]**2 + x[1]**2
    return f

def hsc18(x):
    F = numpy.zeros(2)
    F[0] = x[0]*x[1] - 25
    F[1] = x[0]**2 + x[1]**2 - 25
    F1 = numpy.array( [0, 0] )
    F2 = numpy.array( [numpy.inf, numpy.inf] )
    return F, F1, F2


    
# ---------------------------------------------------------------------
class test_Snobfit(unittest.TestCase):
    def setUp(self):
        pass

    def test_hsf18(self):
        x0 = numpy.array([20, 2])
        u  = numpy.array( [2,0] )
        v  = numpy.array( [50,50] )
        fglob = 5
        xbest, fbest, ncall = snobfit(hsf18, x0, (u, v),
                                      constraintFunc=hsc18,
                                      retall=0,
                                      disp=0,
                                      fglob=fglob
                                      )

       
        assert abs( (fbest-fglob)/fglob ) < 1.0e-2
        


#---------------------------------------------
if __name__ == '__main__':
    unittest.main()
