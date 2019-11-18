import numpy
import pylab
from snobfit.snobfit import snobfit

def ortho_r(n, seed=0):
    numpy.random.seed( [seed] )
    q,r = numpy.linalg.qr( numpy.random.rand( n,n ) )
    return q


class Fm:
    """
    A new class of test functions for global optimization

    Bernardetta Addis, Marco Locatelli

    Equation 6
    """
    def __init__(self,
                 n  = 2,
                 m  = 1,
                 K  = 10.0,
                 H  = 10.0,
                 c1 = -3.0,
                 c2 = 3.0,
                 fac= 0.0,
                 seed=0
                 ):
        self.n=n
        self.m=m
        self.K=K
        self.H=H
        self.c1=c1
        self.c2=c2
        self.c2_c1 = c2 - c1
        self.fac = fac
        self.p=numpy.ones(n)
        self.o = ortho_r(n)
      
    def set_fac(self, fac):
        self.fac = fac
        
    def set_o(self, m):
        self.o=m

    def set_p(self, i):
        if  self.p[i]: self.p[i] = 0
        else:          self.p[i] = 1

        
    def oscillation(self, x):
        """
        Implement Equation(1)
        """
        t = 2.0*numpy.pi * numpy.ceil( (self.K*(self.c2_c1))/10.0 )
        return -self.H*numpy.cos( t * ( (x-self.c1)/(self.c2_c1) ) ) + self.H


    def gamma_p(self, p, x):
        """
        Implement Equation(3)
        """
        if  p==0: return 0.5*(x-self.c2)**2 + 2
        else:     return 0.5*(x-self.c1)**2 + 2


    def getInvA(self, c):
        A = numpy.array( [ [1,  0,   0,     0  ],
                           [1, -c,  c*c, -c*c*c],
                           [0,  1,   0,     0  ],
                           [0,  1, -2*c, 3*c*c ]] )
        return numpy.linalg.inv(A)

        
    def getAlphaBeta(self, p):
        A = self.getInvA(self.c1)            
        b = numpy.array( [ p, 5,0,0] )
        alpha = numpy.dot( A, b)
    
        A = self.getInvA(self.c2) 
        b = numpy.array( [ 1-p, 5,0,0] )
        beta = numpy.dot( A, b)
        
        return ( alpha, beta )


    def beta_p(self, p, x):
        alpha, beta=self.getAlphaBeta( p )

        y = numpy.zeros( len(x) )
        for i in xrange(len(x)):
            if  x[i]<0:
                y[i] = alpha[0] + \
                       alpha[1]*( x[i] - self.c1) + \
                       alpha[2]*( x[i] - self.c1)**2 + \
                       alpha[3]*( x[i] - self.c1)**3
            else:
                y[i] = beta[0] + \
                       beta[1]*( x[i] - self.c2) + \
                       beta[2]*( x[i] - self.c2)**2 + \
                       beta[3]*( x[i] - self.c2)**3
        return y


    def s_pK(self, p, x):
        """
        Implement Equation(2)
        """
        return self.oscillation(x) + self.gamma_p(p, x)


    def d_pK(self, p, x):
        """
        Implement Equation(4)
        """
        return self.oscillation( x) + self.beta_p( p, x)


    def get_wx(self):
        wx = numpy.zeros(self.n)
        for i in xrange(self.n):
            if  (self.p[i]==0 and i<  self.m) or \
                (self.p[i]==1 and i>= self.m) :
                wx[i] = self.c1
            else:
                wx[i] = self.c2

        return wx

    
    def getGlobalMinimizer(self):
        return numpy.dot( numpy.linalg.inv(self.o), self.get_wx() )


    def getGlobalFunc(self):
        return 2*(self.n-self.m)


    
    def __call__(self, x):
        wx = numpy.dot(self.o, x)
        _sum = 0.0
        for i in xrange(self.m):
            _sum += self.d_pK( self.p[i], numpy.array([wx[i]]) )
        for i in xrange(self.m,self.n):
            _sum += self.s_pK( self.p[i], numpy.array([wx[i]]) )

         
        return _sum + self.fac*numpy.random.rand()




#==================================================================
def testsnob4():
    
    u = -5.0 * numpy.ones( 4 )  
    v =  5.0 * numpy.ones( 4 )
    x0 = numpy.array(  [0.7, 4.5, 2.8, -2.4] )
    fglob = 0
    f = Fm(n=4,m=2)
    o = ortho_r(4)
    f.set_o(o)
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc() 
    xbest, fbest, ncall = snobfit(f,
                                  x0,
                                  (u, v),
                                  dn=12,
                                  fglob=fglob,
                                  retall=1
                                  )
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc()     
    print "The best solution", xbest,fbest, ncall

    
def testsnob16():
    
    u = -6.0 * numpy.ones( 16 )  
    v =  6.0 * numpy.ones( 16 )
    x0 = numpy.array(  [0.6, -0.5, 2.8, 0, -0.58, -5.1, -3.3, 4.0,
                        6, 4.3, 2.58, 0.53, -1.6, 1.3, -0.6, -4.3] )
    fglob = 0
    f = Fm(n=16,m=8)
    o = ortho_r(16)
    f.set_o(o)
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc() 
    xbest, fbest, ncall = snobfit(f,
                                  x0,
                                  (u, v),
                                  dn=16,
                                  fglob=fglob,
                                  retall=1
                                  )
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc()     
    print "The best solution", xbest,fbest, ncall

#================================================
def test():
    u = -5.0 * numpy.ones( 2 )  
    v =  5.0 * numpy.ones( 2 )
    x0 = numpy.array(  [0, 1] )
    fglob = 0
    f = Fm()
    f.set_o( -numpy.array( [ [-0.7880367,  -0.61562826],
                             [-0.61562826,  0.7880367 ] ] ) )
    xbest, fbest, ncall = snobfit(f,
                                  x0,
                                  (u, v),
                                  fglob=fglob,
                                  retall=1
                                  )
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc()     
    print "The best solution", xbest,fbest, ncall


def testnoise():
    u = -5.0 * numpy.ones( 2 )  
    v =  5.0 * numpy.ones( 2 )
    x0 = numpy.array(  [0, 1] )
    fglob = 0
    f = Fm()
    f.set_o( -numpy.array( [ [-0.7880367,  -0.61562826],
                             [-0.61562826,  0.7880367 ] ] ) )
    f.set_fac(0.001)
    
    xbest, fbest, ncall = snobfit(f,
                                  x0,
                                  (u, v),
                                  fglob=fglob,
                                  retall=1
                                  )
    
    print "The global minimizer", f.getGlobalMinimizer()
    print "The global function value", f.getGlobalFunc()     
    print "The best solution", xbest,fbest, ncall



if __name__ == '__main__':
    #test()
    testnoise()
