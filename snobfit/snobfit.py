"""
Minimization of a function over a box in R^n.

This software python package SNOBFIT is used to solve the bound constrained
(and soft constrained) noisy optimization of an expensive objective function.
It combines global and local search by branching and local fits.
The package is also made robust and flexible for practical use
by allowing for hidden constraints, batch function evaluations,
change of search regions, etc.

See more details in Paper:
Snobfit - Stable Noisy Optimization by Branch and Fit.
WALTRAUD HUYER and ARNOLD NEUMAIER, Universitat Wien

Usage
-----
Usage:  s = Snobfit( x, f, dx )
        [xbest,fbest, iter] = s.solve( )


History:
-------
1) Ziwen Fu. 10/22/2008
   Directly Translate from snobfit.m ( v2.1 )

2) Ziwen Fu. 11/10/2008
   Python Class version of Snobfit
"""
__docformat__ = "restructuredtext en"

import sys
import numpy

# Import minq  and its subprograms
from minq import minq

# Import util
from util import *

# Import SoftConstraint
from softConstraint import SoftConstraint, SoftConstraints

# TODO List
# 1) Test for more general cases
# 2) Test for big dimension problem

__all__ = ['Snobfit',  # The snobfit class
           'snobfit',  # The snobfit function
           ]

# -----------------------------------------------------------------
class Snobfit:
    """
    Minimization of a function over a box in R^n

    (n = the dimension of the problem)
    """

    # The goldenSection number ( 1/2*[numpy.sqrt(5)-1] )
    goldenSection = 0.618

    def __init__(self,
                 func,
                 x0,
                 bounds,
                 dx=None,
                 constraint=None,
                 p=0.8,
                 dn=5,
                 fglob=None,
                 maxiter=1000,
                 maxfun=1000,
                 disp=0,
                 retall=0,
                 seed=None,
                 xglob=None,
                 nstop=250,
                 fac=0.0,
                 xtol=1.e-6,
                 ftol=1.e-6,
                 rtol=1.e-6,
                 isLeastSqrt = False,
                 retuct=False,
                 callback=None
                 ):
        """  
        Inputs:
        ------
        For properly using Snobfit class, the user must input
        the follow variables:
        func   --  the Python function or method to be minimized.
        x0     --  the initial guess.
        bounds --  the box boundary, it is a list of (lowBounds, highBounds).

        Additional Inputs:
        ------------------
        p      -- probability of generating a evaluation point of Class 4.
        fglob  -- the user specified global function value.
        xglob  -- the user specified global minimum.
        nstop  -- number of times no improvement is tolerated
        rtol   -- a relative error for checking stopping criterion.
        maxiter-- the maximum number of iterations to perform.
        disp   -- non-zero if fval and warnflag outputs are desired.
        retall -- non-zero to return list of solutions at each iteration.

        dx  -- resolution vector, i.e. the i_th coordinate of a point to be
               generated is an integer-valued multiple of dx[i]

               Only used for the definition of a new problem n-vector of
               minimal steps, i.e., two points are considered to be different
               if they differ by at least dx[i] in coordinate i

        dn  -- In the presence of noise, fitting reliable linear models near
               some point requires the use of a few, say dn more data points
               than parameters. ( default 5 )

        seed -- The seed for numpy.random. If the user set this number,it makes
                sure that it returns the same solution from repeated tests

        xtol   -- acceptable relative error in xopt for convergence.
        ftol   -- acceptable relative error in func(xopt) for convergence.
        maxfun -- the maximum number of function evaluations.

        isLeastSqrt -- the minimisation uses the least-squares function or not
        retuct      -- return uncertainty or not

        callback  -- an optional user-supplied function to call after each
                     iteration. It's called as callback(n,xbest,fbest,improved)

        constraint-- an instance of SoftConstraints

        Note:
        -----
        At this stage, we don't implement the code use xtol, ftol, and maxfun.
        Later we should use it.
        """
        # The initial user-supplied guess of the fitting
        self.x0 = x0

        # The dimension of the problem
        self.n = len(self.x0)

        # User defined number
        self.dn = dn

        # Adjust the dn
        self.dn = numpy.maximum(self.dn, self.n/2)
            
        # The suggested number of the evaluation points for fitting.
        self.nreq = self.n + self.dn + 1

        # The number of safeguarded nearest neighbors
        self.snn = self.n + self.dn

        # The Python function or method to be minimized.
        self.func = func

        # The user-defined constraint
        self.constraint = constraint

        # The user defined lower and upper box bounds
        (self.u1, self.v1) = bounds

        # The resolution vector
        self.dx = dx

        # The probability of generating the recommended evaluation points of
        # Class 4
        self.p = p

        # Print flag:  non-zero if fval and warnflag outputs are desired.
        self.disp = disp

        # Return flag: non-zero to return list of solutions at each iteration
        self.retall = retall

        # set the global values
        self.xglob = xglob
        self.fglob = fglob

        # The check stopping criterion (best function value is found
        # with a relative error < rtol
        self.rtol = rtol
        self.ftol = ftol
        self.xtol = xtol

        # The number of times no improvement is tolerated
        self.nstop = nstop

        # The factor for multiplicative perturbation of the data
        self.fac = fac

        # The maximum number of iterations to perform.
        self.maxiter = maxiter

        # The maximum number of function evaluations.
        self.maxfun = maxfun

        # The user-supplied function to call after each iteration.
        self.callback = callback

        # If the minimisation uses the least-squares function
        # eg. f_i = (Y(x, t_i) - y_i) / \sigma_i or not
        self.isLeastSqrt = isLeastSqrt

        # Return the uncertainty the fitting parameters or not?
        self.retuct = retuct

        # Set the seed for numpy.random
        self.seed = seed
        if  self.seed is not None:
            numpy.random.seed( [self.seed] )

        # The rows are a set of new points entering the optimization
        # algorithm together with their function values
        self.x = self._setInitRecommendedEvalPoints( x0 )

        # The corresponding function values ( for self.x )
        # and their uncertainties.
        # a value df[i] <= 0 indicates that the corresponding uncertainty
        # is not known, and the program resets it to numpy.sqrt(eps)
        (self.f, self.df) = self._setInitial_fdf()

        # Current number of the recommended evaluation points
        self.nx = self.x.shape[0]

        # Other initialization
        self.setInit()


    def _setInitRecommendedEvalPoints(self, x0):
        """
        From the initial user-supplied guess, we set up the first set of
        the recommended evaluation points

        Note:
        -----
        Make sure the initial guess x0 is in recommended evaluation points.
        """
        x = numpy.dot( rand(self.nreq-1, self.n),
                       numpy.diag(self.v1-self.u1)
                       ) + \
            numpy.array( [self.u1] ).repeat(self.nreq-1,axis=0)

        x = numpy.append( [x0], x, axis=0 )
        return x


    def _setInitial_fdf(self):
        """
        The computation of the function values of the first set of the
        recommended evaluation function (if necessary, with additive noise)
        """
        f  = vector(self.nreq)
        df = vector(self.nreq)
        if  self.constraint is not None:
            self.F1 = self.constraint.F1()
            self.F2 = self.constraint.F2()
            fm = vector(self.nreq)
            f0 = numpy.inf

        for i in xrange(self.nreq):
            f[i]  = self.func( self.x[i,:] )
            df[i] = max(3*self.fac, numpy.sqrt(eps))

            if  self.constraint is not None:
                FF = self.constraint.F(self.x[i,:])
                if (sum( numpy.logical_and( self.F1<= FF,
                                            FF <= self.F2) ) == self.n ):
                    f0 = min(f0, f[i])


        if  self.constraint is not None:
            Delta = numpy.median( abs(f-f0) )
            for i in xrange(self.nreq):
                fm[i] = self.softmerit(f[i], f0, Delta, self.x[i,:])

            # Save it for later use
            self._f0 = f0
            self._Delta = Delta
            return (fm,df)

        return (f,df)


    def setInit(self):
        """
        Some other initialization of Snobfit.
        """
        if  len(self.f)==0:
            self.f  = vector()
            self.df = vector()

        # If the uncertainty of the function value is negative,
        # set it be the square root of the machine precision.
        ind = find( self.df <= 0 )
        if len(ind) != 0:
            self.df[ind] = numpy.sqrt(eps)

        # Defines the vector of minimal distances between two
        # points suggested in a single call to Snobfit
        self.dy = 0.1*(self.v1-self.u1)

        if  len( find(self.dx==0) ) != 0 :
            raise ValueError('dx should only contain positive entries')

        # ----------------------------------------------------------
        # Some other common data block
        # ----------------------------------------------------------
        # The search bounds [u,v]
        self.u = self.u1.copy()
        self.v = self.v1.copy()

        # vector containing the pointers to the points
        # where the function value could not be obtained
        self.fnan = numpy.array([])

        # near[i,:] is a vector containing the indices of
        # the nearest neighbors of the point x[i,:]
        self.near = self.near = numpy.zeros( (self.nx,self.snn), 'int' )

        # d[j] is maximal distance between x[j,:] and one of its neighbors,
        # or Initial value for the diameter of the trust region
        # for local minimization from the best point
        self.d = inf * numpy.ones(self.nx)

        # small[j] is an logarithmic measure of box j
        self.small = ivector()

        # nsplit[j,i] number of times box j has been split along ith coordinate
        self.nsplit = ivector()

        # nxFval[j] is the number of times the function value
        # of x[j,:] has been measured
        self.nxFval = None

        # t[j] is nxFval[j] times the variance of the function
        # values measured for the point x[j,:]
        self.t = None

        # At the end of fit, the request should be  nreq x (n+2) array
        self.request = numpy.zeros( (0, self.n+2) )

        # Initialize J4, which contains the indices of boxes marked for
        # the generation of the recommended evaluation points point of Class 4,
        # as the empty set.
        self.J4 = ivector()

        # Pointer pointing to the new boxes and
        # boxes whose nearest neighbors have changed
        self.inew = ivector()

        # y, fy are a potential evaluation point and its function values
        self.y  = numpy.zeros( (self.nx, self.n) )
        self.fy = vector( self.nx )

        # The gradient of model
        self.gradient = numpy.zeros( (self.nx, self.n) )

        # The standard deviation of model errors
        self.sigma = vector( self.nx )

        # The follow variables are used to improve the speed.
        self.dx2 = self.dx**2


    def _warning():
        """ Warning message from snobfit """
        raise ValueError("""WARNING: The algorithm was not able to generate \
                 the desired number of points.\n
                 Change the search region or refine resolution vector dx""")
        sys.exit(1)


    def _calc_DMat(self,f):
        """ Compute the D Matrix """
        return numpy.diag( f /self.dx2 )  # self.dx2 = self.dx**2


    def _calc_Smallness(self, xu, xl, v, u ):
        """ Compute the smallness

        This quantity roughly measures how many bisections are
        necessary to obtain this box from [u, v].
        We have S = 0 for the exploration box [u, v],
        and S is large for small boxes.
        For a more thorough global search, boxes with low smallness
        should be preferably explored when selecting new points for evaluation.
        """
        small =- numpy.sum(numpy.round(numpy.log2((xu-xl)/
                           numpy.array([v-u]).repeat(xu.shape[0],axis=0))),
                           axis=1
                           )
        return  numpy.cast['int32']( small )
        

    def _addRequest(self, newReq):
        """ Add a new request. """
        self.request = numpy.append( self.request, newReq, axis=0 )


    def _isMarkedForClass4( self, i, xu, xl, v, u ):
        """
        Decide the index of box ( i.e., i) is marked for the generation of
        the recommended evaluation points of Class 4 or not?
        """
        x = (xu[i,:]-xl[i,:])/(v-u)
        if  numpy.min(x) <= 0.05*numpy.max(x):
            return True
        else:
            return False


    def _sumFind(self, x, n=0, axis=1):
        """ A helper function """
        return find( numpy.sum(x, axis=axis) == n )


    def _maxmin(self, v):
        """ A helper function """
        return ( max(v), min(v) )


    def _estimate(self, f_k, df_k, x_k, x, g, sigma_k):
        """ Calculate the Formula (2) of SNOBFIT by Arnold Neumaier

        fk - f = g^T (x_k-x) + \
                 sigma_k*( (x_k-x)^T D (x_k-x) + df_k ), k = 1, ..., n+dn,

        Input:
        ------
        f_k      --  the function value at x_k
        df_k     --  the uncertainties in the function values f_k
        x_k      --  the nearest neighbors of x, k = 1, . . . , n + dn
        x        --  point x (  a local model around it )
        g        --  the gradient
        sigma_k  --  the model errors,  k = 1, . . . , n+dn

        Output:
        -------
        f -- the estimated function value at x
        """
        y = x_k - x
        D = self._calc_DMat( df_k )
        f = f_k + numpy.dot(y, g) + sigma_k*( dot3(y, D, y.T) + df_k )
        return float(f)


    def _quadEstimate(self, f_best, x_k, x_best, g, G):
        """ Calculate the Formula (3) of SNOBFIT by Arnold Neumaier
        
        f_k - f_best = g^T s_k + 1/2 (s_k)^T G s^k

        Input:
        ------
        f_best -- the function value at the best point x_best
        x_k,   -- the points in X closest to but distinct from x_best.
                  k = 1, . . . ,K
        x_best -- the best point
        g      -- the gradient
        G      -- the symmetric matrix

        Output:
        -------
        f -- the estimated function value at x_k
        """
        s_k = x_k - x_best
        f = f_best + numpy.dot(s_k, g) + 0.5*dot3(s_k, G, s_k.T)
        return float(f)


    def newRequest(self, x, f, i):
        _req = numpy.array([])
        _req = numpy.append( _req, x)
        _req = numpy.append( _req, f)
        _req = numpy.append( _req, i)
        return _req


    def round(self, x, u, v ):
        """
        A point x is projected into the interior of [u,v] and x[i] is
        rounded to the nearest integer multiple of self.dx[i]

        Input:
        ------
        x -- vector of length n
        u -- lower bounds of x
        v -- upper bounds of x
        
        Output:
        -------
        x -- the projected and rounded version of x
        """
        x = numpy.minimum(numpy.maximum(x,u),v)
        x = numpy.round(x/self.dx)*self.dx

        i1 = find(x<u)
        i2 = find(x>v)
        if len(i1) != 0: x[i1] += self.dx[i1]
        if len(i2) != 0: x[i2] -= self.dx[i2]

        return x


    def rsort(self, x):
        """
        Sort x in increasing order, remove multiple entries.

        Warning: when you use this function, make sure x and w is row vector
        """
        x = numpy.sort(x)

        # Remove repetitions
        n    = len(x)
        xnew = numpy.append( x[1:n], inf )
        ind  = find( xnew != x )
        x    = x[ind]

        return x


    # --------------------------------------------------
    def quadMin(self, a, b, xl, xu ):
        """
        Minimization of the quadratic polynomial
        p(x) = a*x^2+b*x over [xl,xu]
        
        Input:
        ------
        a, b,  -- the coefficients of the  quadratic polynomial
        xl, xu -- the bounds (xl < xu)

        Output:
        -------
        x -- the minimizer in [xl,xu]
        """
        if  a > 0:
            x = -0.5*b/a
            x = numpy.minimum( numpy.maximum(xl,x), xu )

        else:
            fl = a*xl**2 + b*xl
            fu = a*xu**2 + b*xu
            if fu <= fl:
                x = xu
            else:
                x = xl

        return float(x)


    def _newBounds(self, xnew, xl, xu, u, v, u1,v1 ):
        """
        If xnew contains points outside the box [u,v], the box is made larger
        such that it just contains all new points; moreover, if necessary,
        [u,v] is made larger to contain [u1,v1]
        the box bounds xl and xu of the boxes on the boundary and the volume
        measures small of these boxes are updated if necessary

        Input:
        ------
        xnew  -- the rows of xnew contain the new points
        xl    -- xl[j,:] is the lower bound of box j
        xu    -- xu[j,:] is the upper bound of box j
        u, v  -- old box bounds
        u1,v1 -- box in which the points are to be generated

        Output:
        -------
        xl   -- updated version of xl
        xu   -- updated version of xu
        u,v  -- updated box bounds (the old box is contained in the new one)

        And:
        small -- updated version of small
        """
        (nx,n) = xl.shape
        uold = u.copy()
        vold = v.copy()

        u = numpy.minimum(numpy.min(xnew, axis=0), u, u1)
        v = numpy.maximum(numpy.max(xnew, axis=0), v, v1)

        i1 = find(u<uold)
        i2 = find(v>vold)

        ind = numpy.array([])
        for j in xrange( len(i1) ):
            j1  = find(xl[:,i1[j]]==uold[i1[j]])
            ind = numpy.append(ind, j1)
            xl[j1,i1[j]] = u[i1[j]]

        for j in xrange( len(i2) ):
            j2  = find(xu[:,i2[j]] == vold[i2[j]] )
            ind = numpy.append(ind, j2)
            xu[j2,i2[j]] = v[ i2[j] ]

        if len(i1) + len(i2):  # At least one of the bounds was changed
            self.small = self._calc_Smallness( xu, xl, v, u )

        return  (xl, xu, u, v)


    def _selectIndexByVariance(self, x ):
        """
        Choose the index i such that the variance of
        x[:,i]/(self.v[i]-self.u[i]) is maxmal.
        """
        variance = numpy.zeros(self.n)
        for i in xrange(self.n):
            variance[i] = std( x[:,i]/(self.v[i]-self.u[i]) )
        return numpy.argmax( variance )


    def _calc_Lambda(self, f1, f2):
        """
        Calculate the lambda for splitting the box
        """
        if  f1 <= f2:
            return self.goldenSection
        else:
            return 1.0 - self.goldenSection


    def _calc_SplitPoint(self, j, x1, x2):
        """
        Compute the split point
        """
        _lambda = self._calc_Lambda(self.func(self.x[j,:]),
                                    self.func(self.x[j+1,:])
                                    )
        return _lambda*x1 + (1.0 - _lambda)*x2


    def _split(self, x, f, df, xl0, xu0, nspl):
        """ Split a box
        
        Splits a box [xl0,xu0] contained in a bigger box [u,v] such that each
        of the resulting boxes contains just one point of a given set of points

        Input:
        ------
        x    -- the rows are a set of points.
        f    -- f[j] contains the function value.
        df   -- its variation.
        xl0  -- vector of lower bounds of the box.
        xu0  -- vector of upper bounds of the box.
        nspl -- nspl[i] is the number of splits the box [xl0,xu0].
                has already undergone along the ith coordinate.
                    
        Output:
        ------
        xl    -- xl[j,:] is the lower bound of box j
        xu    -- xu[j,:] is the upper bound of box j
        x     -- x[j,:]  is the point contained in box j
        f     -- f[j]  contains the function value at x[j,:].
        df    -- df[j] the uncertainty of f[j].
        nsplit-- nsplit[j,i] is the number of times box j has been split
                 in the i-th coordinate
        small -- small[j] is an integer-valued logarithmic volume measure of
                 box j
        """
        (nx,n) = x.shape
        if  nx == 1:
            small = self._calc_Smallness( xu0, xl0, self.v, self.u )
            return (xl, xu, x, f, df, nspl, small )

        elif nx == 2:
            i = numpy.argmax( abs(x[0,:]-x[1,:]) / (self.v-self.u) )

            if self.constraint is None:
                _lambda = self._calc_Lambda(f[0], f[1])
                ymid = _lambda*x[0,i] + (1.0 - _lambda)*x[1,i]
            else:
                ymid = 0.5*(x[0,i] + x[1,i])

            xl = numpy.zeros( (2,len(xl0) ) )
            xu = numpy.zeros( (2,len(xu0) ) )

            xl[0,:] = xl0;  xl[1,:] = xl0
            xu[0,:] = xu0;  xu[1,:] = xu0

            if x[0,i] < x[1,i]:
                xu[0,i] = ymid;  xl[1,i] = ymid
            else:
                xl[0,i] = ymid;  xu[1,i] = ymid

            nsplit = numpy.zeros( (2, len(nspl) ), 'int' )
            nsplit[0,:]  = nspl
            nsplit[0,i] += 1
            nsplit[1,:]  = nsplit[0,:]
            small = self._calc_Smallness( xu, xl, self.v, self.u )

            return (xl, xu, x, f, df, nsplit, small )


        # Choose the index i such that the variance of
        # x[:,i]/(self.v[i]-self.u[i]) is maxmal.
        i = self._selectIndexByVariance( x )

        # Sort the points
        y = self.rsort( x[:,i] )

        j = numpy.argmax( y[1:len(y)]-y[0:len(y)-1] )
        if self.constraint is None:
           ymid = self._calc_SplitPoint(j, y[j], y[j+1])
        else:
           ymid = 0.5*(y[j]+y[j+1])

        ind1 = find( x[:,i]<ymid )
        ind2 = find( x[:,i]>ymid )

        xl = numpy.zeros( (2,len(xl0)) )
        xu = numpy.zeros( (2,len(xu0)) )
        xl[0,:] = xl0; xl[1,:] = xl0;  xl[1,i] = ymid
        xu[0,:] = xu0; xu[1,:] = xu0;  xu[0,i] = ymid

        nsplit = numpy.zeros(  (2,len(xl0)), 'int' )
        nsplit[0,:]  = nspl
        nsplit[0,i] += 1
        nsplit[1,:]  = nsplit[0,:]

        npoint = numpy.array( [len(ind1), len(ind2)] )
        ind = -1*numpy.ones( (2, max(len(ind1),len(ind2))),'int' )
        ind[ 0,0:len(ind1) ] = ind1
        ind[ 1,0:len(ind2) ] = ind2

        nboxes = 1
        [maxpoint,j] = max_(npoint)
        while  int(maxpoint) > 1:

            ind0 = ind[j,find(ind[j,:]>=0)]

            # Choose the index i such that the variance of
            # x[ind0:,i]/(self.v[i]-self.u[i]) is maxmal.
            i = self._selectIndexByVariance( x[ind0,:] )

            # Sort the points
            y = self.rsort( x[ind0,i] )
            d = y[ 1:len(y) ] - y[ 0:len(y)-1 ]
            if len(d) ==0:
                break

            k = numpy.argmax( d )
            if self.constraint is None:
               ymid = self._calc_SplitPoint(k, y[k], y[k+1])
            else:
               ymid = 0.5*(y[k] + y[k+1])

            ind1 = ind0[ find( x[ind0,i]<ymid ) ]
            ind2 = ind0[ find( x[ind0,i]>ymid ) ]

            nboxes += 1

            xl = numpy.append(xl, [ xl[j,:] ], axis=0 )
            xu = numpy.append(xu, [ xu[j,:] ], axis=0 )
            xu[j,i]      = ymid
            xl[nboxes,i] = ymid

            nsplit[j,i] += 1
            nsplit = numpy.append(nsplit, [ nsplit[j,:] ], axis=0 )
            npoint[j] = len(ind1)
            npoint = numpy.append(npoint, len(ind2) )

            ind[j,0:ind.shape[1] ] = -1
            ind[j,0:len(ind1)    ] = ind1
            ind = numpy.append(ind,[-1*numpy.ones(ind.shape[1],'int')], axis=0)
            ind[nboxes,0:len(ind2)] = ind2

            [maxpoint,j] = max_(npoint)

        # ----------------------------------
        x  =  x[ ind[:,0], :]
        f  =  f[ ind[:,0] ]
        df = df[ ind[:,0] ]
        small = self._calc_Smallness( xu, xl, self.v, self.u )

        return ( xl, xu, x, f, df, nsplit, small )


    def _notnanMaxMin(self, notnan):
        """ This function should be merged latter """
        if  len(notnan) !=0 :
            fmin = min( self.f[notnan] )
            fmax = max( self.f[notnan] )
        else:
            fmin = 1
            fmax = 0
        return (fmax,fmin)


    def _nanMaxMin(self):
        """
        Find the max and min function values, but ignore nan elements
        """
        fmin = numpy.nanmin(self.f)
        fmax = numpy.nanmax(self.f)
        return (fmax,fmin)


    def _filterInput(self, x, f, df):
        """
        Checks whether there are any duplicates among the points given by the
        rows of x, throws away the duplicates and computes their average
        function values and an estimated uncertainty

        See details in Algorithm Step 2

        Inputs:
        -------
        x  -- the rows of x are a set of points
        f  -- f[j,:] is function value of x[j,:]
        df -- its uncertainty
                     
        Output:
        ------
        x  -- updated version of x (possibly some points have been deleted)
        f  -- updated version of f (  the average function value ).
        df -- updated version of df(the estimated uncertainty pertaining to x )
        nxFval  -- nxFval[j] is the number of times the row x[j,:] appeared
                   in the input version of x
        t       -- t[j] is nxFval[j] times the variance of the function values
                   measured for point x[j,:]
        """
        (nx,n) = x.shape
        # define np and t
        np = ivector(nx)
        t  = ivector(nx)

        i = 0
        while i < nx:
            j = i+1
            ind = []
            while j < nx:
                if sum( x[i,:]-x[j,:] ) == 0:
                    ind.append(j)
                j += 1

            if  len(ind) !=0 :
                ind  = ind.insert(0,i)
                ind1 = find( numpy.isnan( f[ind,0] ) )

                if len(ind1) < len(ind):
                    ind[ind1] = []
                    np[i] = len( ind )
                    fbar  = sum( f[ind])/np[i]
                    t[i]  = sum( (f[ind]-fbar)**2 )
                    f[i]  = fbar
                    df[i] = numpy.sqrt( (sum( df[i]**2 ) + t[i])/np[i] )
                else:
                    np[i] = 1
                    t[i]  = 0

                # More test here
                x[ind[1:end],:] = []
                f[ind[1:end]]   = []
                df[ind[1:end]]  = []
                nx = x.shape[0]
            else:
                np[i] = 1
                t[i]  = 0

            i += 1

        return [x,f,df,np,t]


    def _calcSnnByInd(self, i):
        """ Computes safeguarded nearest neighbors

        Computes a safeguarded set of nearest neighbors to a point
        x0 = self.x[i,:] for each coordinate, the nearest point differing
        from x0 in the i-th coordinate by at least 1 is chosen
        self.x are the points from which the neighbors are to
        be chosen (possibly x0 is among these points)

        Input:
        ------
        i -- The index of the point around which the fit is to be computed

        Output:
        -------
        near -- Vector pointing to the nearest neighbors(i.e. to the rows of x)
        d    -- Maximal distance between x0 and a point of the set of
                nearest neighbors
        """
        x0 = self.x[i,:] # Point for which the neighbors are to be found
        d = numpy.sqrt( numpy.sum( (self.x-
                                    numpy.array([x0]).repeat(self.nx,axis=0)
                                    )**2,
                                   axis=1
                                   )
                        )
        [d1,ind] = sort(d)

        # Eliminate x0 if it is in the set
        if  not d1[0]:
            ind = numpy.delete(ind, 0)

        near = ivector()
        for i in xrange(self.n):
            j = min(find(abs(self.x[ind,i]-x0[i])-self.dx[i] >= 0))
            near = numpy.append(near, ind[j])
            ind  = numpy.delete(ind, j)

        j = 0
        while near.size < self.snn: # snn: The number of neighbors to be found
            if len(find(near==ind[j]))==0 and \
               max( abs(x0-self.x[ind[j],:]) - self.dx ) >= 0:
               near = numpy.append( near, ind[j] )
            j +=1

        [d,ind] = sort( d[near] )

        return ( near[ind], max(d) )


    def _updateFromNewPoints(self,
                             xl,   # lower bound of the old boxes
                                   # its variation and other parameters
                             xu,   # upper bounds of the old boxes
                                   # its variation and other parameters
                             xnew, # rows contain new points
                             fnew, # new function values and
                             dfnew # their variations
                             ):
        """
        Updates the box parameters when a set of new points and their function
        values are added, i.e., the boxes containing more than one point are
        split and the nearest neighbors are computed or updated.

        Note:
        -----
        self.u1,
        self.v1 -- box in which the new points are to be generated
        self.x  -- rows contain the old points
        self.f  -- f[j] contains the function value at x[j,:]

        Output:
        -------
        xl,xu	-- updated version of xl,xu (including new boxes)
        inew	-- pointer pointing to the new boxes and boxes whose
                   nearest neighbors have changed

        And possibly updated version of:
        self.u, self.v( updated box bounds such that all new points are in box)
        self.fnan --   updated version of fnan (if a function value
                       was found for a point in the new iteration)
        self.near, self.small, self.d, self.nsplit
        self.nxFval,   self.t,     self.x, self.f
        """
        # The number of points from the previous iteration
        nxold = self.nx
        nxnew = xnew.shape[0]
        inew  = ivector()

        # Do we need Case nxold==0???
        # If any of the new points are already among old points, they are
        # thrown away and function value and its uncertainty are updated
        _del = ivector()
        for j in xrange(nxnew):

            i=self._sumFind(abs(numpy.array([xnew[j,:]]).repeat(nxold,axis=0)-
                                self.x),
                            0
                            )
            if  len(i) != 0:

                if  len( find(self.fnan==i) )==0 and \
                       numpy.isfinite( self.f[i] ) :

                    if  not numpy.isnan( fnew[j] ):
                        self.nxFval[i]  += 1
                        delta = fnew[j] - self.f[i]
                        self.f[i] += delta/self.nxFval[i]
                        self.t[i] += delta*(fnew[j]-self.f[i])
                        self.df[i] = numpy.sqrt( self.df[i]**2 + \
                                     ( delta*(fnew[j]-self.f[i])+ \
                                     fnew[j]**2-self.df[i]**2)/self.nxFval[i]
                                     )
                        inew = numpy.append(inew,i)

                else:  # point i had NaN function value

                    if  not numpy.isnan( fnew[j] ):
                        self.f[i]  =  fnew[j]
                        self.df[i] = dfnew[j]
                        inew = numpy.append(inew,i)
                        if len(self.fnan) != 0:
                            ii = find(self.fnan==i)
                            self.fnan = removeByInd(self.fnan, ii)

                _del = numpy.append( _del,  j )


        if  len(_del) != 0 :
            xnew  = numpy.delete( xnew, _del, axis=0 )
            fnew  = numpy.delete( fnew, _del, axis=0 )
            dfnew = numpy.delete(dfnew, _del, axis=0 )

        nxnew = xnew.shape[0]
        if  nxnew==0:
            inew = sort(inew)
            return  (xl, xu, inew)

        # Filter the new recommended evaluation points
        [xnew, fnew0, dfnew0, npnew, tnew] = \
               self._filterInput(xnew, fnew, dfnew )
        nxnew = xnew.shape[0]

        # update current number of points
        self.nx = nxold + nxnew

        # update the new bounds if necessary
        if  numpy.sum(numpy.minimum(numpy.min(xnew,axis=0),self.u)<self.u) or \
            numpy.sum(numpy.maximum(numpy.max(xnew,axis=0),self.v)>self.v) or \
            numpy.sum( numpy.minimum(self.u,self.u1) < self.u ) or \
            numpy.sum( numpy.maximum(self.v,self.v1) > self.v ):
            [xl, xu, self.u, self.v] = \
            self._newBounds(xnew, xl, xu, self.u, self.v, self.u1, self.v1)

        self.x = numpy.append(self.x, xnew, axis=0)
        inew   = numpy.append(inew, range(nxold,self.nx) )

        for i in range(nxold,self.nx):
            self.f  = numpy.append(self.f,   fnew0[i-nxold] )
            self.df = numpy.append(self.df, dfnew0[i-nxold] )

        self.nxFval = numpy.append(self.nxFval, npnew)
        self.t      = numpy.append(self.t,      tnew)

        par = ivector( nxnew )
        for j in xrange(nxnew):
            ind=self._sumFind(within(xl,
                                     numpy.array([xnew[j,:]]).repeat(nxold,
                                                                     axis=0),
                                     xu),
                                 self.n
                                 )
            if  len(ind) !=0 :
                par[j] = ind[ numpy.argmin(self.small[ind]) ]

        par1 = self.rsort(par)
        inew = numpy.append(inew, par1 )

        # Add more space for xl and xu. Check the size is nxnew ???
        xl = numpy.append(xl,
                          numpy.array([vector(self.n)]).repeat(nxnew, axis=0),
                          axis=0
                          )
        xu = numpy.append(xu,
                          numpy.array([vector(self.n)]).repeat(nxnew, axis=0),
                          axis=0
                          )
        self.nsplit=numpy.append(self.nsplit,
                                 numpy.array([ivector(self.n)]).repeat(nxnew,
                                                                       axis=0),
                                 axis=0
                                 )
        self.small =numpy.append(self.small, numpy.zeros(nxnew) )
        for l in xrange( len(par1) ):
            j   = par1[l]
            ind = find(par==j) + nxold
            idx = numpy.append( numpy.array([j]), ind, axis=0 )
            [xl0, xu0, x0, f0, df0, nsplit0, small0] = \
                           self._split(self.x[idx,:],
                                       self.f[idx],
                                       self.df[idx],
                                       xl[j,:],
                                       xu[j,:],
                                       self.nsplit[j,:]
                                       )
            k = self._sumFind( x0 == \
                               numpy.array([self.x[j,:]]).repeat(
                                                           x0.shape[0],axis=0),
                               self.n
                               )
            if len(k) > 1:
                k = k[0]    # Choose the first one

            xl[j,:] = xl0[k,:]
            xu[j,:] = xu0[k,:]
            self.nsplit[j,:] = nsplit0[k,:]
            self.small[j]    = small0[ int(k) ]

            for k in xrange(len(ind)): # number of pts in box [xl[j,:],xu[j,:]]
                k1 = ind[k]
                k2 = self._sumFind(x0 == \
                                   numpy.array([self.x[k1,:]]).repeat(
                                                           x0.shape[0],axis=0),
                                   self.n
                                   )
                if len(k2) > 1:
                    k2 = k2[ numpy.argmin(self.small[k2]) ]

                # More test Later
                xl[k1,:] = xl0[k2,:]
                xu[k1,:] = xu0[k2,:]
                self.nsplit[k1,:] = nsplit0[k2,:]
                self.small[k1]    = small0[k2]

        # Find the current max and min function values.
        (fmax, fmin ) = self._nanMaxMin()

        if  self.nx >= self.snn+1 and fmin < fmax:
            for j in range(nxold,self.nx):
                [nn,dd] = self._calcSnnByInd( j )
                self.d    = numpy.append(self.d, float(dd) )
                self.near = numpy.append(self.near, [nn], axis=0 )

            for j in xrange(nxold):
                xx = ( numpy.array([self.x[j,]]).repeat(nxnew,axis=0)-xnew )**2
                if min( numpy.sqrt( numpy.sum( xx, axis=1 ) ) ) < self.d[j]:
                    [ self.near[j,:], self.d[j] ] = self._calcSnnByInd( j )
                    inew = numpy.append(inew, j)

            inew = self.rsort(inew)
        else:
            self.near = ivector()
            self.d    = inf*ones(1,self.nx)

        return (xl, xu, inew)



    # -----------------------------------------------------------------------
    def _replace_nan(self,
                     inew  # vector pointing to the new boxes and boxes
                           # whose nearest neighbors have changed
                     ):
        """Replaces the function values NaN
        
        Replaces the function values NaN of a set of points
        by a value determined by their nearest neighbors with finite function
        values, with a safeguard for the case that all neighbors
        have function value NaN
        """
        notnan = range(1, self.f.size)
        if len(self.fnan):
            notnan[self.fnan] = []

        [fmax,imax] = max_( self.f )
        fmin = min(self.f)

        dfmax = self.df[imax]

        for j in range( len(self.fnan) ):

            l = self.fnan[j]
            if  len(find(inew==l)) != 0:
                # A substitute function value is only computed for new points
                # and for points whose function values have changed
                ind = self.near[l,:]

                # Eliminate neighbors with function value NaN
                for i in xrange( len(ind) ):
                    if  len( find(self.fnan==ind[i]) ) != 0:
                        ind = numpy.delete( ind, numpy.s_[ i ], axis=0 )

                # Updated self.f
                if  len(ind)==0:
                    self.f[l]  = fmax + 1.e-3*(fmax-fmin)
                    self.df[l] = dfmax
                else:
                    [fmax1,k] = numpy.max( self.f[ind] )
                    fmin1 = numpy.min( self.f[ind] )
                    self.f[l]  = fmax1 + 1.e-3*(fmax1-fmin1)
                    self.df[l] = self.df[ ind[k] ]



    # ------------------------------------------------------------------------
    def _genClass5Points(self, x, nreq):
        """
        Generates nreq recommended evalution points of Class 5
        in [self.u1, self.v1]

        [self.u,self.v]: Bounds of the box in which the points to be generated

        Input:
        ------
        x    -- The points chosen by Snobfit (can be empty)
        nreq -- The number of points to be generated

        Output:
        -------
        x1 -- The rows are the requested points
              x1 is of dimension nreq x n (n = dimension of the problem )
        """
        nx1  = 100*nreq

        xnew = numpy.round( numpy.array([self.u1]).repeat(nx1, axis=0) + \
                            rand(nx1,self.n)* \
                            numpy.array([self.v1-self.u1]).repeat(nx1, axis=0)
                            )
        nx = x.shape[0]
        if  nx:
            d = vector(nx1)
            for j in xrange(nx1):
                xnew[j,:] = self.round(xnew[j,:], self.u1, self.v1)
                d[j]=numpy.min(numpy.sum((x - numpy.array([xnew[j,:]]
                                              ).repeat(nx, axis=0) )**2,
                                         axis=0)
                               )
            ind = find( d==0 )

            if len(ind) != 0:
                xnew[ind,:] = []   # Fix later
                d[ind] = []

            x1  = numpy.array( [] )
            nx1 = xnew.shape[0]

        else:  # TEST Later
            x1    = xnew[0,:]
            xnew  = numpy.delete( xnew, numpy.s_[ 0 ], axis=0 )
            nx1  -= 1
            d = numpy.sum( (xnew - numpy.array([x1]).repeat(nx1, axis=0) )**2,
                           axis=0
                          )
            nreq -= 1

        for j in xrange(nreq):
            i = numpy.argmax(d)
            y = xnew[i,:]
            if j == 0 and len(x1)==0:
                x1 = numpy.mat([y])
            else:
                x1 = numpy.append(x1, [y], axis=0)

            xnew = numpy.delete( xnew, numpy.s_[ i ], axis=0 )
            d    = numpy.delete( d,    numpy.s_[ i ], axis=0 )
            nx1 -= 1

            d1 = numpy.sum( ( xnew-numpy.array([y]).repeat(nx1, axis=0) )**2,
                            axis=1
                            )
            d  = numpy.minimum( d,d1 )

        return numpy.asarray(x1)


    #------------------------------------
    # Step 1
    #------------------------------------
    def update_uv(self):
        """ Update the search bounds [self.u, self.v]

        The search box [self.u, self.v] is updated if the user changed it.
        The vectors self.u and self.v are defined such that [self.u, self.v]
        is the smallest box containing [u', v'], all new input points and,
        in the case of a continuation call to Snobfit, also the box [u1, v1]
        from the previous iteration.
        [self.u, self.v] is considered to be the box to be explored and
        a box tree of [self.u, self.v] is generated;

        Note:
        -----
        All suggested new evaluation points are in [u', v'].
        """
        if  len(self.x) != 0:
            self.u = numpy.minimum( numpy.min(self.x,axis=0), self.u1 )
            self.v = numpy.maximum( numpy.max(self.x,axis=0), self.v1 )


    #------------------------------------
    # Step 2
    #------------------------------------
    def update_input(self):
        """
        Throw out duplicates among the points and
        compute the mean function value and deviation ( ie., self.f, self.df )
        """
        (self.x, self.f, self.df, self.nxFval, self.t) = \
                 self._filterInput(self.x, self.f, self.df)


    #------------------------------------
    # Step 3
    #------------------------------------
    def branch(self):
        """
        All current boxes containing more than one point are split
        according to the algorithm described in Section 5.
        The smallness (also defined in Section 5) is computed for these boxes.
        If [self.u, self.v] is larger than [uold, vold] in a continuation call,
        the box bounds and the smallness are updated for the boxes
        for which this is necessary.
        """
        if  self.nx:
            [self.xl,self.xu,self.x,self.f,self.df,self.nsplit,self.small] = \
                      self._split(self.x,
                                  self.f,
                                  self.df,
                                  self.u,
                                  self.v,
                                  numpy.zeros( self.n )
                                  )
        else:
            self.xl     = vector()
            self.xu     = vector()
            self.nsplit = ivector()
            self.small  = ivector()


    #------------------------------------
    # Step 4
    #------------------------------------
    def calc_snn(self):
        """ Compute the safeguarded nearest neighbors.

        If we have |X| < n + dn + 1 for the current set X of evaluation points,
        or if all function values != NaN (if any) are identical, go to Step 11.
        Otherwise, for each new point x a vector pointing to n + dn
        safeguarded nearest neighbors is computed. The neighbor lists of
        some old points are updated in the same way if necessary.

        Note:
        ----
        We denote the X the set of points for which the objective
        function has already been evaluated at some stage of SNOBFIT.
        """
        for i in xrange( self.nx ):
            [ self.near[i,:], self.d[i] ] = self._calcSnnByInd(i)


    #------------------------------------
    # Step 5
    #------------------------------------
    def calc_fdf_nan(self):
        """ Calculate the fictitious values f and df for nan points

        For any new input point x with function value NaN (which means that
        the function value could not be determined at that point) and for all
        old points marked infeasible whose nearest neighbors have changed,
        we define fictitious values f and df as specified in Section 3.

        Later we test an objective function with nan values
        """
        self.fnan = find( isnan( self.f ) )
        if  not isEmpty(self.fnan):
            self._replace_nan( range(0,self.nx) )


    def _fitLocalModel(self, j ):
        """ Perform a local fit
        
        Computes a local fit around the point x0 = x[j,:]
        and minimizes it on a trust region

        See details in Algorithm Step 6

        Input:
        -------
        j -- The index of the point around which the fit is to be computed

        Output:
        -------
        y    -- Estimated minimizer in the trust region
        fy   -- Its estimated function value
        g    -- Estimated gradient for the fit
        sigma-- sigma = norm(A*g-b)/sqrt(K-n),
                where A and b are the coefficients resp. right hand side of the
                fit, n is the dimension and K the number of nearest neighbors
                considered(estimated standard deviation of the model errors)
        """
        (x0, f0, df0) = ( self.x[j,:], self.f[j], self.df[j] )
        D  = self._calc_DMat(df0)
        x1 = self.x[ self.near[j,:],:]
        K  = x1.shape[0]
        S  = x1 - numpy.array( [x0] ).repeat(K, axis=0)

        d = numpy.maximum(0.5*numpy.max(abs(S), axis=0), self.dx)

        # Set the Q_k = (x_k-x) * D * (x _k -x )
        Q =  vector(K)
        for k in xrange(K):
            Q[k] = dot3( S[k,:],D,toCol(S[k,:]) ) + self.df[ self.near[j,k] ]

        # Now the A_ki = (x_i-x_i^k)/Q_k
        A = S / numpy.array( toCol(Q) ).repeat(self.n, axis=1)

        # Now the b_k = (f-f_k) /Q_k
        b = toCol ( ( self.f[ self.near[j,:] ] - f0) / Q )

        # Reduced SVD
        [U, Sigma, V] = svd(A)

        # To enforce a condition number < 10^-4
        Sigma = numpy.maximum(Sigma, 1.e-4*Sigma[0] )

        # To calc  the Estimated gradient
        g = dot3( numpy.dot(V, numpy.diag( 1.0 / Sigma )), U.T, b)

        # Calc the estimated standard deviation of the model errors
        residual = numpy.array( (A*toCol(g) - b) ) **2
        sigma    = numpy.sqrt( numpy.sum( residual )/(K-self.n) )

        # In the trust region,  calc the Estimated minimizer
        pl = numpy.maximum(-d, self.u-x0)  # lower bounds of p
        pu = numpy.minimum( d, self.v-x0)  # upper bounds of p
        p  = vector(self.n)                # p = y-x0
        for i in xrange(self.n):
            p[i] = self.quadMin( float(sigma*D[i,i]), g[i], pl[i], pu[i] )

        y = self.round(x0+p, self.u, self.v)

        nc = 0
        while min(numpy.max(abs(self.x - \
                                numpy.array([y]).repeat(self.nx, axis=0)) - \
                            numpy.array([self.dx]).repeat(self.nx, axis=0),
                            axis=1))<0  and nc < 5:
            p = pl + (pu-pl)*rand(self.n)
            y = self.round(x0+p, self.u, self.v)
            nc += 1

        # Calc the Estimated Function Value
        fy = self._estimate(self.f[j], self.df[j], y, x0, g, sigma)

        # Return g as row vector
        g  = toRow(g)

        return (y, fy, g, sigma)


    #------------------------------------
    # Step 6
    #------------------------------------
    def localFit(self):
        """ Create local surrogate models.

        Local surrogate models are created by linear least squares fit.
        Local fits (as described in Section 3) around all new points and
        all old points with changed nearest neighbors are computed and
        the potential points y of Class 2 and 3 and their estimated
        function values fy are determined as in Section 4.
        """
        for j in xrange(self.nx):
            self.y[j,:], self.fy[j], self.gradient[j,:], self.sigma[j] = \
                  self._fitLocalModel( j )


    def _nanRequest(self, a):
        b = numpy.array([nan,5]).repeat(self.nreq, axis=0)
        # ( [nan, 5], self.nreq )
        return numpy.append(a,b, axis=1)


    def returnNanRequest(self):
        # We don't test this  part carefully
        # Test this part latter. Fix me latter
        x5 = self._genClass5Points(self.x, self.nreq)
        self.request = self._nanRequest(x5)
        if  len(self.x) != 0 :
            [ fbest, ind] = min_( self.f )
            xbest = self.x[ind,:]

        else:
            xbest = nan*numpy.ones(self.n)
            fbest = inf

            if  self._isNotEnoughReq():
                self._warning()

        return (self.request, xbest, fbest )


    def _extend(self, a, n):
        """
        Extend the Array a about n element(s).
        """
        if len(a.shape)==1: nadd = numpy.zeros( n )
        else:               nadd = numpy.zeros( (n, a.shape[1]) )
        return numpy.append(a, numpy.array(nadd), axis=0)


    def updateWorkspace(self):
        """ update the current work space """
        oldxbest = self.xbest

        [self.xl, self.xu, self.inew] = \
                 self._updateFromNewPoints(self.xl,
                                           self.xu,
                                           self.xnew,
                                           self.fnew,
                                           self.dfnew
                                           )

        if  not isEmpty(self.near):
            ind = find( isnan(self.f) )
            self.fnan = numpy.append(self.fnan, ind)
            if  not isEmpty(self.fnan):
                self._replace_nan( self.inew )

            # Extend more space for y, g, and sigma.
            nadd = max(self.inew) - self.y.shape[0] + 1
            if nadd > 0:
                self.y        = self._extend( self.y,        nadd )
                self.fy       = self._extend( self.fy,       nadd )
                self.sigma    = self._extend( self.sigma,    nadd )
                self.gradient = self._extend( self.gradient, nadd )

            for j in self.inew:
                self.y[j], self.fy[j], self.gradient[j,:], self.sigma[j] = \
                           self._fitLocalModel(j)



    # -----------------------------------------------------------------
    def _quadFit(self, j ):
       """ A quadratic model around the best point is fitted and minimized
       with minq over a trust region
       
       Input:
       ------
       j -- the index of the best point

       Output:
       -------
       y  -- minimizer of the quadratic model around the best point
       fy -- its estimated function value
       """
       (x0, f0) = (self.x[j,:], self.f[j])

       K = min( self.nx-1, self.n*(self.n+3) )
       # K = min( self.nx-1, self.n*(self.n+3)/2 + self.dn )
       distance=numpy.sum((self.x-numpy.array([x0]).repeat(self.nx,axis=0))**2,
                           axis=1
                           )
       ind = numpy.argsort(distance)[1:K+1]

       d = numpy.max(abs( self.x[self.near[j,:],:] - \
                          numpy.array([x0]).repeat(self.snn, axis=0) ),
                     axis=0
                     )
       d = numpy.maximum(d, self.dx)

       # S^k = x^k - x0 ( xbest )
       S = self.x[ind,:] - numpy.array( [x0] ).repeat(K, axis=0)

       R = numpy.linalg.qr(S, mode='r')
       
       # When we meet the sigular system, just reject it,
       # and return current best solution.
       try:
           L = inv(R).T
       except:
           return (self.xbest, self.fbest)

       
       # Scaling Matrix H which esures affine invariance of fitting procedure
       # Note: 1) S^T * H * S = || R^-T * S ||^2
       #       2) The exponent 3/2 in the error term reflects the expected
       #          cubic approx. order of the quadratic model
       sc = numpy.sum( numpy.dot(S,L.T)**2, axis=1) ** (3.0/2.0 )
       b = (self.f[ind]-f0)/sc

       xx = self.x[ind,:] - numpy.array( [x0] ).repeat(K, axis=0)
       xx2 = 0.5*xx*xx
       A = numpy.bmat( 'xx xx2')

       for i in xrange(self.n-1):
           xx = (self.x[ind,i] - x0[i])
           B  = numpy.array( toCol(xx) ).repeat( self.n-i-1, axis=1)
           xx = B*(self.x[ind,i+1:self.n] - \
                   numpy.array( [x0[i+1:self.n]] ).repeat(K, axis=0) )
           A = numpy.bmat('A xx')

       xx = numpy.array( toCol(sc) ).repeat(A.shape[1], axis=1)
       A = A/xx

       # In q, it includes the components of g and the upper triangle of G
       q = mldiv(A,toCol(b))

       # G is a symmetric matrix
       G = numpy.diag( q[self.n:2*self.n] )
       k = 2*self.n
       for i in xrange( self.n-1 ):
           for j in xrange(self.n-i-1):
               G[i, j+i+1] = q[k]
               G[j+i+1, i] = q[k]
               k += 1

       # The estimate gradient
       g = q[0:self.n]

       # Solve the minimizer of the quadratic model around the best point x0
       if x0.shape[0] == 1:
          y = self.quadMin( float(G),
                            float(g-numpy.dot(G,x0)),
                            float(numpy.maximum(x0-d, self.u1)),
                            float(numpy.minimum(x0+d, self.v1))
                            ) 
       else:
           [y,info] = minq(g-numpy.dot(G, x0),
                           G,
                           numpy.maximum(x0-d, self.u1),
                           numpy.minimum(x0+d, self.v1),
                           )
       y = self.round(y, self.u1, self.v1)

       nc = 0
       while nc <10 and \
             min(numpy.max( abs( self.x - \
                                 numpy.array([y]).repeat(self.nx,axis=0) ) - \
                            numpy.array( [self.dx] ).repeat(self.nx, axis=0),
                            axis=1))<-eps:

           u1 = numpy.maximum(x0-d, self.u1)
           v1 = numpy.minimum(x0+d, self.v1)

           y = u1 + numpy.dot( rand(1,self.n), (v1-u1) )
           y = self.round(y, self.u1, self.v1)
           nc += 1

       fy = self._quadEstimate(f0, y, x0, g, G)

       return (y, fy )


    #------------------------------------
    # Step 7
    #------------------------------------
    def locQuadFit(self):
        """ Do a local quadratic fit
        
        The current best point xbest and the current best function value fbest
        in [self.u1, self.v1] are determined. If the objective function
        has not been evaluated in [self.u1, self.v1] yet, (i.e., n1 = 0),
        it mean that we doesn't generate the recommended evalution points of
        Class 1. And go to Step 8.

        A local quadratic fit around self.xbest is computed as described
        in Section 3 and the point z of Class 1 is generated as described
        in Section 4. If such a point z was generated, let z be contained in
        the subbox [\underline{x}^j, \bar{x}^j ] (in the case that z belongs
        to more than one box, a box with minimal smallness is selected).
        if
                     min_i( (\bar{x}^j_i - \underline{x}^j_i )/(v_i-u_i) ) >
              0.05 * max_i( (\bar{x}^j_i - \underline{x}^j_i )/(v_i-u_i) ),

        the point z is put into the list of recommended evaluation points,
        and otherwise (if the smallest side length of the box
        [ \bar{x}_j , \underline{x}_j ] relative to [self.u, self.v] is
        too small compared to the largest one) we set self.J4 = {j}, i.e.,
        a point of Class 4 is to be generated in this box later,
        which seems to be more appropriate for such a long, narrow box.
        This gives n1 recommended evaluation points (n1 = 0 or 1) of Class 1.
        """
        if  numpy.sum( within(self.u1,self.xbest,self.v1), axis=0) == self.n:
            j_Class1 = self.jbest
        else:
            j_Class1 = self.ind[ numpy.argmin(self.f[self.ind]) ]

        [z, fz] = self._quadFit( j_Class1 )
        z = self.round(z, self.u1, self.v1)

        j = self._sumFind( within(self.xl,
                                  numpy.array([z]).repeat(self.nx, axis=0),
                                  self.xu),
                           self.n
                           )
        if  len(j) > 1:
            j = j[ numpy.argmin(self.small[j]) ]

        if  min( numpy.max(abs(self.x- \
                               numpy.array( [z] ).repeat(self.nx, axis=0) ) - \
                           numpy.array( [self.dx] ).repeat(self.nx, axis=0),
                           axis=1) ) >= -eps:

            if  self._isMarkedForClass4(j,self.xu,self.xl,self.v,self.u):
                self.J4 = numpy.append(self.J4, j)
            else:
                self.request = numpy.array( [ self.newRequest(z,fz,1) ] )


    #------------------------------------
    # Step 8
    #------------------------------------
    def assignN4EachClass(self):
        """ Use random number to assign new evaluation points for each Class.

        To generate the remaining self.globloc := nreq - n1 recommended
        evaluation points, let glob1 := floor(self.p*self.globloc)
                               glob2 := ceil*(self.p*self.globloc).

        Then a random number generator sets self.glob = glob1 with
        probability self.globloc*self.p - glob1 and self.glob=glob2 otherwise.
        Then self.globloc - self.glob points of Classes 2 and 3 together
        and self.glob points of Class 4 are to be generated.
        """
        # the remaining recommended evaluation points
        self.globloc = self.nreq - self.request.shape[0]

        glob1 = self.globloc * self.p
        glob2 = int( numpy.floor(glob1) )
        if rand() < glob1 - glob2:  self.glob = glob2 + 1
        else:                       self.glob = glob2


    def _genPoints(self, k, local, classID = 2):
        """ a Helper Function """
        j = 0
        sreq = self.request.shape[0]
        while sreq < self.nreq-self.glob and j < len(local):

            i = local[ k[j] ]
            y = self.round( self.y[i,:], self.u1, self.v1)

            l = self._sumFind( within(self.xl,
                                      numpy.array([y]).repeat(self.nx,axis=0),
                                      self.xu),
                               self.n
                               )

            if len(l) == 0:   # To solve the empty bug
               j += 1
               continue

            if len(l) > 1:
                l = l[ numpy.argmin(self.small[l]) ]

            if  self._isMarkedForClass4(l, self.xu,self.xl, self.v, self.u):
                self.J4 = numpy.append(self.J4, l)
                j += 1
                continue

            yy  = self.request[:,0:self.n] - \
                  numpy.array([y]).repeat(sreq, axis=0)
            dxy = numpy.array( [numpy.maximum(self.dy,self.dx)]
                              ).repeat(sreq, axis = 0)
            if  numpy.max( abs(y-self.x[l,:])-self.dx) >= -eps and \
                ( not sreq or \
                  min( numpy.max(abs(yy)-dxy, axis=1) ) >= -eps):

                if sum(y==self.y[i,:]) < self.n:
                    f = self._estimate(self.f[i],
                                       self.df[i],
                                       y,
                                       self.x[i,:],
                                       self.gradient[i,:],
                                       self.sigma[i]
                                       )
                else:
                    f = self.fy[i]

                self._addRequest( [self.newRequest(y, f, classID)] )

            sreq = self.request.shape[0]
            j += 1



    # ---------------------------------------------------------------------
    def _calcLocalPoints(self,
                         ind=None 
                         ):
        """
        Computes a pointer to all `local' points (i.e. points whose neighbors
        have `significantly larger' function values)

        Input:
        -------
        ind -- pointer to the boxes to be considered

        Output:
        -------
        local    -- vector containing the indices of all local points
        nonLocal -- vector containing the indices of all nonlocal points
        """
        if  ind==None:
            ind = range(0,self.nx)

        jj       = ivector()
        local    = ivector()
        nonLocal = ind

        for j in xrange( len(ind) ):
            ( fmax,fmin ) = self._maxmin( self.f[ self.near[ind[j],:] ] )
            if  self.f[ ind[j] ] < fmin-0.2*(fmax-fmin):
                local = numpy.append(local, ind[j] )
                jj    = numpy.append(jj, j)

        if  len(jj) != 0:
            nonLocal = removeByInd( nonLocal, jj )

        return (local,nonLocal)


    def _constructPoints(self, i):
        """
        For a box [self.xl, self.xu] containing a point x, a point y in the
        intersection of [self.xl, self.xu] and [self.u1, self.v1] is
        constructed such that it is both not close to x and to the boundary
        of [self.xl,self.xu] and its function value is estimated
        from a local quadratic model around x

        The local quadratic model around x is given by

            q(y) = self.f[i] + self.gradient*(y-x)' + \
                   self.sigma*( (y-x)*diag(D)*(y-x)' + self.df[i] )

            For a row vector y, where D = df0 / dx**2

        Input:
        ------
        i -- the index of the point around which the fit is to be computed

        Output:
        -------
        y  -- Point in the intersection of [self.xl,self.xu] and
              [self.u1, self.v1]
        fy -- Corresponding estimated function value
        """
        # Class 4.
        y = vector(self.n)
        for j in xrange(self.n):
            if  self.x[i,j]-self.xl[i,j] > self.xu[i,j]-self.x[i,j]:
                y[j] = 0.5*(self.xl[i,j] + self.x[i,j] )
            else:
                y[j] = 0.5*(self.x[ i,j] + self.xu[i,j] )

        y = self.round(numpy.minimum( numpy.maximum(y,self.u1),self.v1),
                       numpy.maximum(self.xl[i,:], self.u1),
                       numpy.minimum(self.xu[i,:], self.v1)
                       )
        fy = self._estimate(self.f[i],
                            self.df[i],
                            y,
                            self.x[i],
                            self.gradient[i,:],
                            self.sigma[i]
                           )
        return (y,fy)


    #------------------------------------
    # Step 9
    #------------------------------------
    def genPointsFromClass234(self):
        """  Generate the new points for Class 2,3,4

        If X contains any local points, first at most

        loc := self.globloc - self.glob

        points generated from the local points y are chosen in the order of
        ascending model function values fy to yield points of Class 2.

        If the desired number of loc points has not been reached yet,
        afterwards points y pertaining to nonlocal x which belong to X are
        taken (again in the order of ascending fy).
        For each potential point of Class 2 or 3 generated in this way,
        a subbox [ \underline{x}^j , \bar{x}^j ] of the box tree with
        minimal smallness is determined with
        y belong to  [ \underline{x}^j , \bar{x}^j ]
        if
                     min_i( (\bar{x}^j_i - \underline{x}^j_i )/(v_i-u_i) ) >
              0.05 * max_i( (\bar{x}^j_i - \underline{x}^j_i )/(v_i-u_i) ),

        the point y is not put into the list of recommended evaluation points
        but instead we set self.J4 = self.J4 and {j}, i.e., the box is marked
        for the generation of a recommended evaluation point of Class 4.

        Note: we denote the X the set of points for which the objective
              function has already been evaluated at some stage of SNOBFIT.
        """
        loc = self.globloc - self.glob
        if  loc:
            [local,nonlocal] =  self._calcLocalPoints( self.ind )
            k = numpy.argsort( self.fy[local] )
            self._genPoints(k, local, classID=2)

            if  self.request.shape[0] < self.nreq - self.glob:
                k = numpy.argsort( self.fy[nonlocal] )
                self._genPoints(k, nonlocal, classID=3)

        #---------------------------------------------------
        # There are some suggested evaluation points which don't put into list
        # Here we put it into list with Class 4
        sreq = self.request.shape[0]
        for i in self.J4:
            self.ind = removeByInd( self.ind, find(self.ind==i) )

            [y, fy] = self._constructPoints(i)

            yy = self.request[:,0:self.n] - \
                 numpy.array([y]).repeat(sreq, axis=0)
            if  numpy.max( abs( y-self.x[i,:])-self.dx) >= -eps and \
                ( not sreq or \
                  min( numpy.max(abs(yy) - \
                                 numpy.array([self.dx]).repeat(sreq, axis=0),
                                 axis=1) ) >= -eps):

                self._addRequest( [ self.newRequest(y, fy, 4) ] )

            sreq = self.request.shape[0]
            if  sreq == self.nreq:
                break

    #------------------------------------
    # Step 10
    #------------------------------------
    def genClass4_FromSmall(self):
        """
        Let Smin and Smax denote the minimal and  maximal smallness,
        respectively, among the boxes in current box tree, and let

          M := floor(1/3(Smax-Smin)).

        For each smallness S = Smin + m, m = 0, . . . ,M, the boxes are sorted
        according to ascending function values f(x) (i.e., self.f( self.x) ).
        First a point of Class 4 is generated from the box with S = Smin with
        smallest f(x). If self.J4 != Null, then the points of Class 4
        belonging to the subboxes with indices in self.J4 are generated.
        Subsequently, a point of Class 4 is generated in the box with
        smallest f(x) at each nonempty smallness level Smin+m, m = 1, ... ,M,
        and then the smallness levels from Smin to Smin+M are gone through
        again etc. until we either have nreq recommended evaluation points
        or there are no eligible boxes for generating points of Class 4
        any more. We assign to the points of Class 4 the model function values
        obtained from the local models pertaining to the points in their boxes.
        """
        first = 1
        sreq = self.request.shape[0]
        (Smax,Smin) = self._maxmin( self.small[self.ind] )
        M = int( numpy.floor( (Smax-Smin)/3.0 ) )
        while sreq < self.nreq and len(self.ind) != 0:
            for m in xrange(M+1):

                if  first == 1:
                    first = 0
                    continue

                m = 0
                k = find( self.small[self.ind] == Smin+m )
                while len(k)==0:
                    m += 1
                    k  = find( self.small[self.ind] == Smin+m )

                if  len(k) !=0 :
                    k = self.ind[k]
                    k  = k[ numpy.argsort(self.f[k]) ]
                    i  = k[0]
                    self.ind = removeByInd( self.ind,  find(self.ind==i) )

                    [y, fy] = self._constructPoints(i)

                    yy  = self.request[:,0:self.n] - \
                          numpy.array( [y] ).repeat(sreq, axis=0)
                    dxy = numpy.array([numpy.maximum(self.dy, self.dx)]
                                      ).repeat(sreq, axis=0)
                    if  numpy.max( abs(y-self.x[i,:])-self.dx) >= -eps and \
                        ( not sreq or \
                        min( numpy.max(abs(yy)-dxy, axis=1) ) >= -eps):

                        self._addRequest( [self.newRequest(y, fy, 4)] )

                    sreq = self.request.shape[0]
                    if sreq == self.nreq:
                        break

                m = 0


    #------------------------------------
    # Step 11
    #------------------------------------
    def genPointsFromClass5(self):
        """
        If the number of the recommended evaluation points is still
        less than self.nreq, the set of evaluation points is filled up
        with points of Class 5 as described in Section 4.
        If local models are already available, the model function values
        for the points of Class 5 are determined as the ones for the points
        of Class 4 (see Step 10); otherwise, they are set to NaN.
        """
        x1 = self._genClass5Points(
                   numpy.append( self.x, self.request[:,0:self.n], axis=0 ),
                   self.nreq - self.request.shape[0]
                   )
        for j in xrange( x1.shape[0] ):

            i=self._sumFind(within(self.xl,
                                  numpy.array([x1[j:]]).repeat(self.nx,axis=0),
                                   self.xu),
                            self.n
                            )
            if  len(i) > 1:
                i = i[ numpy.argmin(self.small[i]) ]

            f = self._estimate(self.f[i],
                               self.df[i],
                               x1[j,:],
                               self.x[i,:],
                               self.gradient[i,:],
                               self.sigma[i]
                               )

            self._addRequest([ self.newRequest(x1[j,:],f,5) ])



    def _prepare(self):
        """ Some preparation work
        1) Init request
        2) Make sure that the first a point of Class 4 is generated
           from the box with S = Smin with smallest f(x).
        etc.
        """
        self.request = numpy.zeros( (0, self.n+2) )

        # Make sure that lower and upper bounds of the search boxes
        # to belong to the user's specified bounds at begining.
        xx = numpy.logical_and( self.xl <= \
                                numpy.array([self.v1]).repeat(self.nx,axis=0),
                                self.xu >= \
                                numpy.array([self.u1]).repeat(self.nx,axis=0) )
        self.ind = find( numpy.sum( xx, axis=1 ) == self.n )

        # Find the first index of the recommended eval pts of Class 4
        k = find( self.small[self.ind] == min(self.small[self.ind]) )
        k = k[ numpy.argsort(self.f[ self.ind[k] ]) ]
        self.J4 = numpy.array( [k[0]] )

        # The curent best solution
        [self.fbest, self.jbest] = min_( self.f )
        ind = numpy.argsort(self.f)
        self.xbest = self.x[self.jbest,:]


    def _getExtrmeFunc(self):
        notnan = find( numpy.isfinite( self.f ) )
        return self._notnanMaxMin(notnan)


    def _isNotEnoughReq(self):
        """
        If there is enough request return True.
        Otherwise return False
        """
        if  self.request.shape[0] < self.nreq:
            return True
        else:
            return False


    def _q(self, f, f0, Delta):
        return ( f - f0 ) / ( Delta + abs(f - f0) )

    def softmerit(self, f, f0, Delta, x, Fx=None):
        """
        Merit function of the soft optimality theorem

        Input:
        ------
        f     -- objective function value
        f0    -- scalar parameter in the merit function
        Delta -- scalar, positive parameter in the merit function
        x     -- the variable.
        """
        if Fx is not None:
           _Fx = Fx
        else:
           _Fx = self.constraint.F(x)

        if  not numpy.isfinite(f) or \
            not numpy.any( numpy.isfinite( _Fx ) ):
            # If the objective function or one of the constraint functions is
            # infinite or NaN, set the merit function value to 3
            return 3

        return self._q(f, f0, Delta) + self.constraint.r(x)


    def softmeritOLD(self, f,F,F1,F2,f0,Delta,sigma):
        """
        Merit function of the soft optimality theorem

        Input:
        ------
        f     -- objective function value
        F     -- vector containing the values of the constraint functions
              -- (m-vector)
        F1    -- m-vector of lower bounds of the constraints
        F2    -- m-vector of upper bounds of the constraints
        f0    -- scalar parameter in the merit function
        Delta -- scalar, positive parameter in the merit function
        sigma -- positive m-vector, where sigma(i) is the permitted violation
                 of constraint i
        """
        if  not numpy.isfinite(f) or \
            not numpy.any( numpy.isfinite(F) ):
            # If the objective function or one of the constraint functions is
            # infinite or NaN, set the merit function value to 3
            fm = 3
            return fm

        m = len(F)
        delta = 0
        for i in xrange(m):
            if   F[i] < F1[i]:
                 delta += (F1[i]-  F[i])**2/sigma[i]**2

            elif F[i] > F2[i]:
                 delta += (F[i] - F2[i])**2/sigma[i]**2

        fm = self._q(f, f0, Delta) + 2*delta/(1+delta)
        return fm


    # -----------------------------------------------------------------
    def _calcCovarAtSolution(self, x, f, df=None):
        """ calc the Correlation Matrix or Uncertainty at the solution

        OutPut:
        -------
        covar   the Correlation Matrix.
        """
        if x  is not None: self.xnew  = x.copy()
        if f  is not None: self.fnew  = f.copy()
        if df is not None:
            self.dfnew = df.copy()
        else:
            self.dfnew = numpy.ones(len(f))*max(3*self.fac,numpy.sqrt(eps))

        self.updateWorkspace()

        # Current best solution
        [fSolution, j] = min_( self.f )
        xSolution = self.x[j,:]

        K = min( self.nx-1, self.n*(self.n+3) )
        distance=numpy.sum((self.x - \
                           numpy.array([xSolution]).repeat(self.nx,axis=0))**2,
                           axis=1
                           )
        ind = numpy.argsort(distance)[1:K+1]

        # S^k = x^k - xSolution
        S = self.x[ind,:] - numpy.array([xSolution]).repeat(K,axis=0)

        R = numpy.linalg.qr(a, mode='r')
        L = inv(R).T

        # Scale matrix
        sc = numpy.sum( numpy.dot(S,L.T)**2, axis=1) ** (3.0/2.0 )
        b = (self.f[ind]-fSolution)/sc

        xx  = self.x[ind,:] - numpy.array([xSolution]).repeat(K,axis=0)
        xx2 = 0.5*xx*xx
        A   = numpy.bmat( 'xx xx2')

        for i in xrange(self.n-1):
            xx =( self.x[ind,i] - xSolution[i])
            B  = numpy.array( toCol(xx) ).repeat( self.n-i-1, axis=1)
            xx = B*(self.x[ind,i+1:self.n] - \
                    numpy.array([ xSolution[i+1:self.n] ]).repeat(K,axis=0) )
            A  = numpy.bmat('A xx')

        xx = numpy.array( toCol(sc) ).repeat(A.shape[1], axis=1)
        A = A/xx

        # In q, it includes the components of g and the upper triangle of G
        q = mldiv(A, toCol(b))

        # Jacobian matrix
        J = numpy.diag( q[self.n:2*self.n] )/2
        k = 2*self.n
        for i in xrange( self.n-1 ):
           for j in xrange(self.n-i-1):
               J[i, j+i+1] = q[k]/2
               J[j+i+1, i] = q[k]/2
               k += 1

        # The covariance matrix is computed by,
        # covar = (J^T J)^{-1}
        covar = inv(J.T * J)

        # If the minimisation uses the weighted least-squares function.
        # the covariance matrix should be multiplied by the variance of
        # the residuals about the best-solution least-squares.
        # For example, the fit of the reflectometry problem.
        if self.isLeastSqrt:
            # Calc the squared residuals(i.e., the goodness of fit)
            # at the solution.
            sumsq = abs(fSolution)**2/self.n
            covar *= sumsq


        return covar


    def _covarianceMatrix(self):
        """
        Calculate the Covariance Matrix at the solution
        """
        # Compute function values at the suggested points
        nrequest = self.request.shape[0]
        x = numpy.zeros( (nrequest,self.n) )
        f = vector( nrequest )
        for j in xrange( nrequest ):
            x[j,:] = self.request[j,0:self.n]
            f[j]   = self.func(x[j,:])

        return self. _calcCovarAtSolution(x, f)

    # The alias
    covar = _covarianceMatrix


    def uncertainty(self):
        """
        Calculate the uncertainty of the fitting parameter at the solution.
        """
        covar = self._covarianceMatrix()

        # calculate the uncertaity of fit parameters.
        uct = numpy.sqrt( numpy.diag(covar) )

        return uct


    def fit(self, x=None, f=None, df=None, initialCall=False):
        """
        One call of snobfit

        Output:
        -------
        request	nreq x (n+3)-matrix
           request[j,0:n] is the jth newly generated point,
           request[j,n+1] is its estimated function value and
           request[j,n+2] is its uncertainty of estimated function value
           request[j,n+3] indicates for which reason the point
                          = 1 best prediction
                          = 2 putative local minimizer
                          = 3 alternative good point
                          = 4 explore empty region
                          = 5 to fill up the required number of function values
                              if too little points of other classes are found
        xbest  current best point
        fbest  current best function value (i.e. function value at xbest)
        """
        if initialCall:
            # When a new run is started
            self.update_uv()     # Step 1
            self.update_input()  # Step 2
            self.branch()        # step 3

            (fmax, fmin) = self._getExtrmeFunc()
            if  self.nx >= self.snn + 1 and fmin < fmax:
                self.calc_snn()      # Step 4
                self.calc_fdf_nan()  # step 5
                self.localFit()      # Step 6
            else:
                # This is equal go to Step 11
                return self.returnNanRequest()

        else: # a continue call
            # For the continue call, we needn't go through Step 1-6,
            # We just update the work space for last call
            if x  is not None: self.xnew  = x.copy()
            if f  is not None: self.fnew  = f.copy()
            if df is not None:
               self.dfnew = df.copy()
            else:
               self.dfnew = numpy.ones(len(f))*max(3*self.fac,numpy.sqrt(eps))

            self.updateWorkspace()

            if  isEmpty(self.near): # This is equal go to Step 11
                self.returnNanRequest()

        self._prepare()

        self.locQuadFit()   # Step 7

        if  self._isNotEnoughReq():
            self.assignN4EachClass()      # Step 8
            self.genPointsFromClass234()  # step 9

        self.genClass4_FromSmall()        # step 10

        if  self._isNotEnoughReq():
            self.genPointsFromClass5()    # Step 11

        if  self._isNotEnoughReq():
            self._warning()



    def solve(self):
        """ Main Loop

        The optimization is stopped at one of the follow conditions:
        1: if approximately ncall fucntion values have been exceeded.

        2: if at least minfval function values were obtained and
           the best function value wasn't improved in the last nstop
           calls to SNOBFIT

        3: stop when within tolerance of the known global minimum. E.g.,
           for least squares problems, the global minimum is 0 for a
           perfect match.  Various test functions also have known global
           minima.

        Output:
        -------
        xopt  -- the minimizer of function.
        fopt  -- the value of function at minimum: fopt = func(xopt).
        ncall -- the number of iterations.
        """
        # Function call counter
        ncall = self.nreq
        xopt  = inf
        improvement = 0

        # iteration counter
        i = 0

        # Define x and f
        x = numpy.zeros( (self.nreq, self.n) )
        f = numpy.zeros( self.nreq )

        # Repeat until the limit on function calls is reached
        while ncall < self.maxiter:
             if ncall == self.nreq:
                 # initial call
                 self.fit(initialCall=True)

             else:
                 # continuation call
                 self.fit(x=x, f=f)

             [xopt, fopt] = (self.xbest, self.fbest)
             # Compute (perturbed) function values at the suggested points
             for j in xrange( self.nreq ):
                 x[j,:] = self.request[j,0:self.n]
                 f[j]   = self.func(x[j,:])

             # update function call counter
             ncall += self.nreq

             # best of the new function values
             [fbestn, jbest] = min_(f)

             # If a better function value has been found, update fbest
             if fbestn < fopt:
                 fopt = fbestn
                 xopt = x[jbest,:]
                 improvement = 0
             else:
                 improvement += 1

             # Stop if no improvement expected
             if improvement >= self.nstop:
                 break

             if  self.callback is not None :
                 self.callback(i, x[jbest], fbestn, improvement==0)

             # Stop at user specified stopping criterion
             # best function value is found with a relative error < 1.e-6
             if  self.fglob is not None:
                 if self.fglob:
                     if  abs((fopt-self.fglob)/self.fglob) < self.rtol:
                         if self.disp: print("The best solution:", xopt,fopt)
                         break
                 else: # safeguard if functions with fglob=0 are added by user
                     if  abs(fopt) < self.ftol:
                         if self.disp: print("The best results", xopt, fopt)
                         break

             if  self.retall:
                 print(xopt, fopt, ncall)

             i+=1

        if  self.retuct:
            return  xopt, self.uncertainty(), fopt, ncall
        else:
            return  xopt, fopt, ncall


    def softSolve(self):
        """ Main Loop

        The optimization is stopped at one of the follow conditions:
        1: if approximately ncall fucntion values have been exceeded.

        2: if at least minfval function values were obtained and
           the best function value wasn't improved in the last nstop
           calls to SNOBFIT

        3: stop when within tolerance of the known global minimum. E.g.,
           for least squares problems, the global minimum is 0 for a
           perfect match.  Various test functions also have known global
           minima.

        Output:
        -------
        xopt  -- the minimizer of function.
        fopt  -- the value of function at minimum: fopt = func(xopt).
        ncall -- the number of iterations.
        """
        # Function call counter
        ncall = self.nreq
        xopt  = inf
        improvement = 0
        change = 0

        # iteration counter
        i = 0

        # Calc the objective function for the starting points.
        _x = self.x.copy()
        _f = vector( self.nreq )
        for j in xrange( self.nreq ):
          _f[j] = self.func( _x[j] )


        # Calc the constraints for the starting points.
        # get scalar parameter in the merit function
        _F = numpy.zeros( ( self.nreq, len(self.constraint.C) ) )
        f0 = self._f0
        Delta = self._Delta
        for j in xrange( len(self.f) ):
            _F[j] = self.constraint.F(self.x[j,:])

        # Some declaration of vector
        F  = numpy.zeros( (self.nreq, len(self.constraint.C)) )
        f  = numpy.zeros( self.nreq )

        # Two common constants
        sigmaLo = self.constraint.sigmaLo()
        sigmaHi = self.constraint.sigmaHi()

        # Repeat until the limit on function calls is reached
        while ncall < self.maxiter:
             if ncall == self.nreq:
                 # initial call
                 self.fit(initialCall=True)

             else:
                 # continuation call
                 self.fit(x=x, f=fm)

             [xopt, fopt] = (self.xbest, self.fbest)

             # Compute (perturbed) function values at the suggested points
             x  = numpy.zeros( (self.nreq, self.n) )
             fm = numpy.zeros( self.nreq )

             for j in xrange( self.nreq ):
                 x[j,:] = self.request[j,0:self.n]
                 f[j]   = self.func(x[j,:])
                 FF    = self.constraint.F(x[j,:])
                 F[j]  = FF
                 fm[j] = self.softmerit(f[j] , f0, Delta, x[j,:], Fx=FF)

                 if f[j] <= self.fglob  and \
                    min(FF - self.F1 + sigmaLo) >= 0 and \
                    min(self.F2 + sigmaHi  -FF) >= 0:
                    ncall += j
                    xsoft = x[j,:]
                    [fbestn,jbest] = min_(fm)
                    if fbestn < fopt:
                        fopt = fbestn
                        xopt = x[jbest,:]

                    # show number of function values, best point and function
                    # value and the point xsoft fulfilling the conditions of
                    # the soft optimality theorem (hopefully close to xglob)
                    if self.disp:
                       print(ncall,xopt,fopt,xsoft)

             # Append new suggested points
             _x = numpy.append(_x, x, axis=0)
             _f = numpy.append(_f, f, axis=0)
             _F = numpy.append(_F, F, axis=0)

             # update function call counter
             ncall += self.nreq

             # best of the new function values
             [fbestn, jbest] = min_(fm)

             # If a better function value has been found, update fbest
             if fbestn < fopt:
                 fopt = fbestn
                 xopt = x[jbest,:]
                 improvement = 0
             else:
                 improvement += 1

             # Stop if no improvement expected
             if improvement >= self.nstop:
                 break

             if  self.callback is not None :
                 self.callback(i, x[jbest], fbestn, improvement==0)

             if  fopt < 0 and change == 0:
                 n = _x.shape[0]
                 c1 = numpy.min( _F - numpy.array([self.F1]).repeat(n,axis=0),
                                 axis=1 ) > -eps
                 c2 = numpy.min( numpy.array([self.F2]).repeat(n,axis=0)-_F,
                                 axis=1) > -eps
                 ind = find( numpy.logical_and(c1, c2) )
                 if len(ind) != 0 :
                    change = 1
                    f0 = min(_f[ind])
                    Delta = numpy.median( abs(f-f0) )
                    fm = vector( n )
                    for j in xrange(n):
                        fm[j]=self.softmerit(_f[j], f0, Delta, _x[j] )

                 x = _x.copy()

             _fopt = self.func(xopt)

             # Stop at user specified stopping criterion
             # best function value is found with a relative error < 1.e-6
             if  self.fglob is not None:
                 if self.fglob:
                     if  abs((_fopt-self.fglob)/self.fglob) < self.rtol:
                         if self.disp: print("The best solution:", xopt,_fopt)
                         break
                 else: # safeguard if functions with fglob=0 are added by user
                     if  abs(_fopt) < self.ftol:
                         if self.disp: print("The best results", xopt, _fopt)
                         break

             if  self.retall:
                 print(xopt, _fopt, ncall)

             i+=1

        if  self.retuct:
            return  xopt, self.uncertainty(), _fopt, ncall
        else:
            return  xopt, _fopt, ncall


# --------------------------------------------------------------------------
# A scipy type of optimizer which use Snobfit Algorithm.
# The user can use this interface to solve his/her problem.
# -------------------------------------------------------------------------
def snobfit(func,
            x0,
            bounds,
            p      = 0.5,
            dn     = 5,
            xglob  = None,
            fglob  = 0,
            fac    = 0,
            rtol   = 1e-6,
            xtol   = 1e-6,
            ftol   = 1e-6,
            maxiter= 2000,
            maxfun = 2000,
            disp   = 0,
            retall = 0,
            isLeastSqrt = False,
            retuct      = False,
            constraint  = None,
            callback    = None,
            seed        = None
            ):
    """
    Minimize a function using the Snobfit algorithm with
    the box bound constrains.

    Description:
    ------------
    Uses a Snobfit algorithm to find the minimum of function
    of one or more variables with the [low, high] bounds.

    Inputs:
    -------
    For properly use snobfit function, we must input the follow parameters:
    func   -- the Python function or method to be minimized.
    x0     -- the initial guess.
    bounds -- the box boundary, it is a list of (lowBounds, highBounds).

    Outputs:
    --------
    xopt  --  minimizer of function.
    fopt  --  value of function at minimum: fopt = func(xopt).
    ncall --  number of iterations.

    Additional Inputs:
    ------------------
    p       -- probability of generating a evaluation point of Class 4.
    fac     -- Factor for multiplicative perturbation of the data.
    fglob   -- the user specified global function value.
    xglob   -- the user specified global minimum.
    rtol    -- a relative error
    xtol    -- acceptable relative error in xopt for convergence.
    ftol    -- acceptable relative error in func(xopt) for convergence.
    maxiter -- the maximum number of iterations to perform.
    maxfun  -- the maximum number of function evaluations.
    disp    -- non-zero if fval and warnflag outputs are desired.
    retall  -- non-zero to return list of solutions at each iteration.
    callback-- an optional user-supplied function to call after each iteration.
               It is called as callback(n,xbest,fbest,improved)
    isLeastSqrt -- the minimisation uses the least-squares function or not.
    retuct      -- Return the uncertainty the fitting parameters or not?

    Examples
    --------
    >>> from snobfit import snobfit
    >>> def ros(x):
    >>>     f = 100*( x[0]**2 - x[1] ) **2 + (x[0] - 1) **2
    >>>     return fa = eye(4*6)
    >>>
    >>> x0 = numpy.array([2, 3])
    >>> u  = -5.12*numpy.ones(2)
    >>> v = -u
    >>> fglob = 0
    >>> xglob = numpy.array([1, 1])
    >>> xbest,fbest,ncall = snobfit(ros, x0, (u,v),retall=1,disp=1,fglob=0)
    >>> xbest
    >>> (1.0001,1.000002)
    """
    (u, v)= bounds

    # resolution vector
    dx = (v-u)*1.0e-5

    # Snobfit Class
    S = Snobfit( func, x0, bounds,
                 dx=dx,
                 dn=dn,
                 fglob=fglob,
                 constraint = constraint,
                 fac=fac,
                 ftol=ftol,
                 xtol=xtol,
                 rtol=rtol,
                 disp=disp,
                 retall=retall,
                 p=0.5,
                 maxiter=maxiter,
                 maxfun=maxfun,
                 seed=seed,
                 isLeastSqrt=isLeastSqrt,
                 retuct=retuct,
                 callback=callback
                 )
    if  retuct:
        if constraint is None:
            xopt, uct_x, fopt, ncall = S.solve()
        else:
            xopt, uct_x, fopt, ncall = S.softSolve()
        return xopt, uct_x, fopt, ncall
    else:
        if constraint is None:
            xopt, fopt, ncall = S.solve()
        else:
            xopt, fopt, ncall = S.softSolve()

        return xopt, fopt, ncall



# ++++++++++++++++++++++++++++++++++++++++++++++++++++
# A small test
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
def _fx(x0):
    return numpy.sum((x0-5)*(x0-5))

def test():
    x0 = numpy.array([60.0, 6.0])
    xl = numpy.array([0.0, 0.0])
    xh = numpy.array([100.0, 100.0])

    xbest, fbest, ncall = snobfit(_fx, x0, (xl, xh), retall=1)
    print('optimize:',xbest, fbest, ncall)


def ros(x):
    f = 100*( x[0]**2 - x[1] ) **2 + (x[0] - 1) **2
    return f


def test_ros():
    """ Rosenbrock """
    x0 = numpy.array([2, 3])
    u = -5.12*numpy.ones(2)
    v = -u
    fglob = 0
    xglob = numpy.array([1, 1])
    xbest, fbest, ncall = snobfit(ros, x0, (u, v), retall=1, disp=1, fglob=0 )


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

def test_hsf18():
    x0 = numpy.array([30, 4])
    u = numpy.array( [2,0] )
    v = numpy.array( [50,50] )
    fglob = 5

    def hsc18_1(x):
        return x[0]*x[1] - 25

    def hsc18_2(x):
        return x[0]**2 + x[1]**2 - 25

    s = SoftConstraints()
    # ------------------------------------------------------------
    s.add(  SoftConstraint( F=hsc18_1, Flo=0, Fhi=numpy.inf, sigma=1.25) )
    s.add(  SoftConstraint( F=hsc18_2, Flo=0, Fhi=numpy.inf, sigma=1.25) )

    xbest, fbest, ncall = snobfit(hsf18, x0, (u, v),
                                  constraint=s,
                                  retall=1, disp=1, fglob=fglob
                                  )

if __name__ == '__main__':
    #test()
    test_ros()
    #test_hsf18()
