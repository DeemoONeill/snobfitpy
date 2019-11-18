"""
testrefl.py
----------------------------------------------------------------------
Python file for running SNOBFIT on a simple reflectometry code
"""

import numpy
from reflectometry.model1d.model.profile import Profile
from reflectometry.model1d.tests.parratt import refl
from snobfit.snobfit import snobfit

# Model
D2O = 6.3 # x10^-6 inv A^2
Air = 0.0
Si  = 2.07 # x10-6 inv A^2
Au  = 4.7

# Rhos
rhoDopcBest = [-0.584963,    1.45143,  -0.227783,   -1.32106,  -0.341007, 1.03835,  -0.158153 ]

# sigma, mu, depth 
depthDopcBest = [6.97921,    5.75761,    10.1118,    8.50052,    8.02034,    5.15141,  8.38418 ]

# load the dataset
data1 = numpy.loadtxt('qr.si.au.dopc.d2o.42408').T
Q1,R1 = data1

p0 = rhoDopcBest + depthDopcBest


# Profile
def model_profile(p=None):
    if p == None:
        common_rhos=[Si, Au] + rhoDopcBest  
        depths = [17.538,     38.6251] + depthDopcBest + [100]
    else:
        common_rhos=[Si, Au]
        depths = [17.538, 38.6251]
        for i in xrange(len(p)/2):
            common_rhos.append(p[i])
            depths.append( p[len(p)/2+i] )
        depths.append( 100 )

    rhos1  = common_rhos + [ D2O ]
    names1 = ["Si", "Au", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D20" ]
    sigmas = [2.0, 1.521, 2.461, 5.0, 2.05, 2.056, 2.45, 0.789, 0.617]
    m1 = Profile( names=names1,
                  depth=depths,
                  rough=sigmas,
                  rho=rhos1,
                  mu=[0.0]*10
                  )
    z,p1 = m1.calc(n=100)  # ignore mu
    
    return z, p1[0]


def defaults():
    pbest = rhoDopcBest + depthDopcBest
    lo=[]; hi=[]
    for i in xrange(len(pbest)):
        lo.append( pbest[i]-abs(pbest[i])*0.2 )
        hi.append( pbest[i]+abs(pbest[i])*0.2 )
    return (numpy.array(lo), numpy.array(hi), 0.0)


def reflectivity( p ):
    z,p1= model_profile(p)
    thickness = z[1:]-z[:-1]
    rho1 = p1[1:]
    Rcalc1 = abs(refl(Q1,thickness,rho1))**2
    dR1    = numpy.log(Rcalc1) - numpy.log(R1)
    chisq  = numpy.dot(dR1,dR1)/len(Rcalc1) 
    
    return chisq



def snobtest( ):
    (u,v,fglob) = defaults()
    xbest, fbest, ncall = snobfit(reflectivity, p0, (u, v),
                                  dn=10,
                                  maxiter=1000,
                                  retall=True
                                  )

    print xbest,fbest, ncall


#----------------------------------------------
if __name__ == '__main__':
    snobtest()


