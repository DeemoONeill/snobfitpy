""" Unit tests for Snobfit for data with noise
Author: Ziwen Fu
Nov 2007
"""

import unittest
import numpy
try:
    from snobfit.snobfit import snobfit
except:
    from snobfit import snobfit

def ros(x):
    f = 100*( x[0]**2 - x[1] ) **2 + (x[0] - 1) **2
    return f


def bra(x):

    a=1
    b=5.1/(4*numpy.pi*numpy.pi)
    c=5/numpy.pi
    d=6
    h=10
    ff=1/(8*numpy.pi)
    
    x1 = x[0]
    x2 = x[1]

    f = a * (x2-b * x1 **2 + c*x1-d )**2 + \
        h* (1-ff) * numpy.cos(x1) + h

    return f

def cam(x):
    f = ( 4- 2.1 * x[0]**2 + x[0]**4 /3 ) * x[0]**2 + \
        x[0] * x[1] + \
        (-4 + 4.0 * x[1]**2 ) * x[1]**2
    return f

def gpr(x,y=None):
    if  y == None:
        x1 = x[0]
        x2 = x[1]
    else:
        x1 = x
        x2 = y

    f =( 1+ (x1+x2+1)**2 *(19-14*x1+3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**22) ) * \
       ( 30+(2*x1-3*x2)**2 * (18-32*x1 + 12*x1**2 + 48*x2 - \
                              36*x1*x2 + 27*x2**2) )

    return f


def sh5(x):
    a = numpy.array( [[4.0, 1.0, 8.0, 6.0, 3.0],
                      [4.0, 1.0, 8.0, 6.0, 7.0],
                      [4.0, 1.0, 8.0, 6.0, 3.0],
                      [4.0, 1.0, 8.0, 6.0, 7.0]])
    c = numpy.array( [0.1, 0.2, 0.2, 0.4, 0.4] )
    d = numpy.zeros(5)
    for i in xrange(5):
        b = ( x - a[:,i] ) ** 2
        d[i] = sum(b)

    f = -sum( (c+d) ** (-1) )

    return f


def sh7(x):
    a = numpy.array( [[4, 1, 8, 6, 3, 2, 5],
                      [4, 1, 8, 6, 7, 9, 5],
                      [4, 1, 8, 6, 3, 2, 3],
                      [4, 1, 8, 6, 7, 9, 3]]
                      )
    c = numpy.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])
    d =  numpy.zeros(a.shape[1])
    for i in xrange(7):
        b = ( x - a[:,i] ) **2
        d[i] = sum(b)

    f = -sum( (c+d)**(-1) )
    return f


def sh10(x):
    a = numpy.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                     [4, 1, 8, 6, 7, 9, 5, 1, 2, 3.6],
                     [4, 1, 8, 6, 3, 2, 3, 8, 6, 7],
                     [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])
    c = numpy.array( [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5] )

    d = numpy.zeros(10)
    for i in xrange(10):
        b = ( x - a[:,i] ) ** 2
        d[i] = numpy.sum(b)

    f = -numpy.sum( (c+d)**(-1) )
    
    return f


def hm3(x):
    a = numpy.array([[3.0,  0.1,  3.0,  0.1],
                     [10.0, 10.0, 10.0, 10.0],
                     [30.0, 35.0, 30.0, 35.0]])

    p = numpy.array( [[ 0.36890, 0.46990, 0.10910, 0.03815],
                      [ 0.11700, 0.43870, 0.87320, 0.57430],
                      [ 0.26730, 0.74700, 0.55470, 0.88280]])
    
    c = numpy.array( [1.0, 1.2, 3.0, 3.2] )
    d = numpy.zeros(4)
    for i in xrange(4):
        d[i] = sum( a[:,i] * ( x - p[:,i]) **2 )
        
    f = -sum( c * numpy.exp( -d ) )

    return f


def hm6(x):
    a = numpy.array([[ 10.00,  0.05,  3.00, 17.00],
                     [ 3.00, 10.00,  3.50,  8.00],
                     [ 17.00, 17.00,  1.70,  0.05],
                     [ 3.50,  0.10, 10.00, 10.00],
                     [ 1.70,  8.00, 17.00,  0.10],
                     [ 8.00, 14.00,  8.00, 14.00]])
    
    p = numpy.array([[ 0.1312, 0.2329, 0.2348, 0.4047],
                     [ 0.1696, 0.4135, 0.1451, 0.8828],
                     [ 0.5569, 0.8307, 0.3522, 0.8732],
                     [ 0.0124, 0.3736, 0.2883, 0.5743],
                     [ 0.8283, 0.1004, 0.3047, 0.1091],
                     [ 0.5886, 0.9991, 0.6650, 0.0381]])
    
    c = numpy.array( [1.0, 1.2, 3.0, 3.2] )

    d = numpy.zeros(4)
    for i in xrange(4):
        d[i] = numpy.sum( a[:,i] * ( x - p[:,i] ) **2 )

    f = -numpy.sum( c*numpy.exp(-d)); 

    return f

def shu(x):
    # Shubert function
    sum1 = 0
    sum2 = 0
    for i in range(1,6):
        sum1 += i * numpy.cos( (i+1) * x[0] + i )
        sum2 += i * numpy.cos( (i+1) * x[1] + i );

    f = sum1*sum2
    return f


rtol = 1.e-2

# ---------------------------------------------------------------------
class test_Sigma_Case1(unittest.TestCase):

    def setUp(self):
        self.factor=0.0
        print "Sigma=0.0"
        

    def test_bra(self):
        u = numpy.array( [-5, 0])
        v = numpy.array( [10, 15] )
        x0    = numpy.array( [2,3] )
        fglob = 0.397887357729739
        xglob = numpy.array( [[9.42477796, -3.14159265, 3.14159265], 
                              [2.47499998, 12.27500000, 2.27500000]] )

        xbest, fbest, ncall = snobfit(bra, x0, (u, v),
                                      fglob=fglob,
                                      rtol=rtol,
                                      fac = self.factor
                                      )
        assert abs( (fbest-fglob)/fglob)  < rtol
        print "branin", xbest, fbest, ncall

        
    def test_ros(self):
        """ Rosenbrock """
        x0 = numpy.array([2, 3])
        u = -5.12*numpy.ones(2)
        v = -u
        fglob = 0
        xglob = numpy.array([1, 1])
        xbest, fbest, ncall = snobfit(ros, x0, (u, v),
                                      fac = self.factor)
    
        assert abs(fbest) < 1.0e-6
        print "ros: ", xbest, fbest, ncall

        
    def test_cam(self):
        # six-hump camel
        u = numpy.array([-3, -2])
        v = numpy.array([3, 2] )
        x0 = numpy.array([1, 1])
        fglob = -1.0316284535
        xglob = numpy.array([ 0.08984201,  -0.08984201,
                              -0.71265640,   0.71265640])
        xbest, fbest, ncall = snobfit(cam, x0, (u, v), fglob=fglob,rtol=rtol,
                                      fac = self.factor)
    
        assert abs((fbest-fglob)/fglob) < rtol
        print "cam", xbest, fbest, ncall

    def test_sh5(self):
        # Shekel 5
        u = numpy.array([0, 0, 0, 0])
        v = numpy.array([10, 10, 10, 10] )
        x0 = numpy.array([5, 5,5,5])
        fglob = -10.1531996790582
        xglob = numpy.array([4, 4, 4, 4])
        xbest, fbest, ncall = snobfit(sh5, x0, (u, v),
                                      fglob=fglob,rtol=rtol,
                                      fac = self.factor)
    
        assert abs( (fbest-fglob)/fglob ) < rtol
        print "sh5", xbest, fbest, ncall


    def test_sh7(self):
        #Shekel 7
        u = numpy.array([0, 0, 0, 0])
        v = numpy.array([10, 10, 10, 10])
        x0 = numpy.array([5, 5,5,5])
        fglob = -10.4029405668187
        xglob = numpy.array([4, 4, 4, 4])
        xbest, fbest, ncall = snobfit(sh7, x0, (u, v),
                                      fglob=fglob,rtol=rtol,fac = self.factor)
    
        assert abs( (fbest-fglob)/fglob ) < rtol
        print "sh7", xbest, fbest, ncall

        
    def test_sh10(self):
        # Shekel 10
        u = numpy.array([0, 0, 0, 0])
        v = numpy.array([10, 10, 10, 10])
        x0 = numpy.array([5, 5,5,5])
        fglob = -10.5364098166920
        xglob = numpy.array([4, 4, 4, 4])
        xbest, fbest, ncall = snobfit(sh10, x0, (u, v), fglob=fglob,rtol=rtol,
                                      fac = self.factor)
    
        assert abs( (fbest-fglob)/fglob )  < rtol
        print "sh10", xbest, fbest, ncall

        
    def test_hm3(self):
        # Hartman 3
        u = numpy.array([0, 0, 0])
        v = numpy.array([1, 1, 1])
        x0 = numpy.array([0.5, 0.5,0.5])
        fglob = -3.86278214782076
        xglob = numpy.array([0.1, 0.55592003, 0.85218259])
        xbest, fbest, ncall = snobfit(hm3, x0, (u, v), fglob=fglob,rtol=rtol,
                                      fac = self.factor)
    
        assert abs(( fbest-fglob)/fglob ) < rtol
        print "hm3", xbest, fbest, ncall

        
    def test_hm6(self):
        # Hartman 6
        u = numpy.array([0, 0, 0, 0, 0, 0])
        v = numpy.array([1, 1, 1, 1, 1, 1])
        x0 = numpy.array([0.5, 0.5,0.5,0.5,0.5,0.5])
        fglob = -3.32236801141551
        xglob = numpy.array([0.20168952,  0.15001069,  0.47687398,
                             0.27533243,  0.31165162,  0.65730054])
        xbest, fbest, ncall = snobfit(hm6, x0, (u, v), fglob=fglob,rtol=rtol,
                                      fac = self.factor)

        assert abs( (fbest-fglob)/fglob ) < rtol
        print "hm6", xbest, fbest, ncall

        
    def test_shu(self):
        # Shubert
        u = numpy.array([-10, -10])
        v = numpy.array([10, 10])
        x0 = numpy.array([1, 1])
        fglob = -186.730908831024
        xglob = numpy.array([[ -7.08350658,  5.48286415,  4.85805691,
                              4.85805691, -7.08350658, -7.70831382,
                              -1.42512845, -0.80032121, -1.42512844,
                              -7.08350639, -7.70831354,  5.48286415,
                              5.48286415,  4.85805691, -7.70831354,
                              -0.80032121, -1.42512845, -0.80032121],
                            [4.85805691,  4.85805681, -7.08350658,
                            5.48286415, -7.70831382, -7.08350658,
                            -0.80032121, -1.42512845, -7.08350639,
                            -1.42512844,  5.48286415, -7.70831354,
                            4.85805691,  5.48286415, -0.80032121,
                            -7.70831354, -0.80032121, -1.42512845]])

        xbest, fbest, ncall = snobfit(shu, x0, (u, v), fglob=fglob,
                                      fac = self.factor,rtol=rtol
                                      )     
        assert abs( (fbest-fglob)/fglob ) < rtol
        print "shu", xbest, fbest, ncall



# ---------------------------------------------------------------------
class test_Sigma_Case2(test_Sigma_Case1):
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """
    def setUp(self):
        self.factor=0.01
        print "Sigma=0.01"


# ---------------------------------------------------------------------
class test_Sigma_Case3(test_Sigma_Case1):
    """ Test case for a simple constrained entropy maximization problem
    (the machine translation example of Berger et al in
    Computational Linguistics, vol 22, num 1, pp 39--72, 1996.)
    """
    def setUp(self):
        self.factor=0.1
        print "Sigma=0.1"

        
#---------------------------------------------
if __name__ == '__main__':
    unittest.main()
