"""
softConstraint Class

Usage
-----
Usage:  c = softConstraint()
       

History:
-------
1) Ziwen Fu. 01/08/2009

"""
__docformat__ = "restructuredtext en"

import numpy


# -----------------------------------------------------------------
class SoftConstraint:
    """
    SoftConstraint Class
    """
    def __init__(self, F=None, Flo = 0, Fhi = numpy.inf, sigma=1.25):
        self.F = F
        self.Flo = Flo
        self.Fhi = Fhi
        self.sigmaLo = sigma
        self.sigmaHi = sigma

    def delta(self, x):
        Fx = self.F(x)
        if   Fx < self.Flo:
             return  (Fx - self.Flo)/self.sigmaLo
        elif Fx > self.Fhi:
             return  (Fx - self.Fhi)/self.sigmaHi
        else:
             return 0

    def getFx(self,x):
        return self.F(x)

    

# -----------------------------------------------------------------
class SoftConstraints:
    """
    SoftConstraints Class
    """
    def __init__(self):
        self.C = []
        
    def add(self, c):
        self.C.append(c)

    def r(self,x):
        m = len(self.C)
        delta = 0
        for i in xrange(m):
            delta += self.C[i].delta(x)**2
        return delta

    def F(self,x):
        Fs = numpy.zeros( len(self.C) )
        for i in xrange(len(self.C)):
            Fs[i] = self.C[i].getFx(x)
        return Fs

    def F1(self):
        F1s = numpy.zeros( len(self.C) )
        for i in xrange(len(self.C)):
            F1s[i] = self.C[i].Flo
        return F1s

    def F2(self):
        F2s = numpy.zeros( len(self.C) )
        for i in xrange(len(self.C)):
            F2s[i] = self.C[i].Fhi
        return F2s

    def sigmaLo(self):
        s = numpy.zeros( len(self.C) )
        for i in xrange(len(self.C)):
            s[i] = self.C[i].sigmaLo
        return s

    def sigmaHi(self):
        s = numpy.zeros( len(self.C) )
        for i in xrange(len(self.C)):
            s[i] = self.C[i].sigmaHi
        return s


       
# ------------------------------------------------------------
def hsc18_1(x):
    return x[0]*x[1] - 25

def hsc18_2(x):
    return x[0]**2 + x[1]**2 - 25

if __name__ == '__main__':
    s =  SoftConstraints()
    s.add(  SoftConstraint( F=hsc18_1, Flo=0, Fhi=numpy.inf) )
    s.add(  SoftConstraint( F=hsc18_2, Flo=0, Fhi=numpy.inf) )
    print s.C
    print len(s.C)
    print s.r( [1,1] )
    print s.F( [1,1] )
    print s.F1( )
    print s.F2( )
