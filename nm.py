import numpy as np
from functools import reduce
import bisect
import math
from random import random as rand

class util:
    #product of function for vals in iterable
    def product(function, iterable):
        return reduce(lambda x,y: x * function(y), iterable, 1)
    
    #sum of function for vals in iterable
    def _sum(function, iterable):
        return reduce(lambda x,y: x + function(y), iterable, 0)

class interpolation:
    def lagrange_interpolate(xs, ys, val = None):
        if len(xs) == len(ys):
            N = len(xs) - 1
        else:
            raise Exception("Interpolation unequal number of x and y points")
        #multiplies a degree N polynomial by a linear polynomial, modded by x^n
        def lin_poly_mult(norm_poly, lin_poly):
            alpha = lin_poly[1]
            beta = lin_poly[0]
            return [beta * norm_poly[i] + alpha * norm_poly[i-1] if i > 0 else 
                    beta * norm_poly[0] for i in range(len(norm_poly))]
        if not val is None:
            #find interpolation value at a particular point
            if val in xs:
                return ys[xs.index(val)]
            w = [] #weights of the lagrange interpolation in barycentric coords
            for j in range(N + 1):
               cur_x = xs[j]
               w_a = util.product(lambda l: cur_x - xs[l], range(j))
               w_b = util.product(lambda l: cur_x - xs[l], range(j+1, N+1))
               w_rev = w_a * w_b
               if w_rev == 0:
                   raise Exception("Interpolation with repeated x value: "
                                   + str(round(cur_x, 5)))
               cur_w = 1.0 / (w_rev)
               w.append(cur_w)
            numerator = 0
            denominator = 0
            for j in range(N+1):
                v = w[j] / (val - xs[j])
                numerator += v * ys[j]
                denominator += v
            if denominator == 0:
                #shouldn't happen
                raise Exception("Interpolation Failed")
            else:
                return numerator / denominator
        else:
            #find coeficients of interpolating polynomial
            #we simply use Sylvester's formula
            coefs = [0] * (N+1)
            for j in range(N+1):
                lagrange_j = [1] + [0] * N
                for k in range(N+1):
                    if k == j:
                        continue
                    diff = xs[j] - xs[k]
                    if diff == 0:
                        raise Exception("Interpolation with repeated x value: "
                                   + str(round(xs[j], 5)))
                    lagrange_j = lin_poly_mult(lagrange_j, 
                                               [-xs[k] / diff, 1/diff])
                for i in range(N+1):
                    coefs[i] += lagrange_j[i] * ys[j]
            return coefs
                
    #interpolate using a legendre basis, just returns coefficients
    def legendre_basis_polynomial(xs, ys):
        if len(xs) == len(ys):
            N = len(xs) - 1
        else:
            raise Exception("Interpolation unequal number of x and y points")
        if len(xs) != len(set(xs)):
            raise Exception("Interpolation with repeated x value")
        #use dp to compute matrix
        A = np.zeros((N+1, N+1))
        for i in range(N+1):
            A[i, 0] = 1
            A[i, 1] = xs[i]
        for i in range(N+1):
            for j in range(2, N+1):
                A[i, j] = ((2*j-1)*xs[i]*A[i, j-1]-(j-1)*A[i, j-2]) / (j)
        #solve the system to get the coefficients
        coeffs = np.linalg.solve(A, ys)
        return coeffs
    
    #interpolate using cubic splines. Assumes xs is sorted.
    def cubic_spline_interpolate(xs, ys, derivs, val = None):
        if not (len(xs) == len(ys) and len(xs) == len(derivs)):
            raise Exception("Interpolation unequal number of x,y,y' points")
        def get_coefs(x0, x1, y0, y1, v0, v1):
            #simple solve a linear system for the coefficients
            #the system for a spline can classically be reduced to a
            #tridiagonal one, but because the dimension is low the speed
            #improvement is not only neglible but detrimental
            A = np.array([[x0 ** 3, x0 ** 2, x0, 1],
                          [x1 ** 3, x1 ** 2, x1, 1],
                          [3 * x0**2, 2 * x0, 1, 0],
                          [3 * x1**2, 2 * x1, 1, 0]])
            b = np.array([y0, y1, v0, v1])
            return np.linalg.solve(A, b)
        if not val is None:
            #binary search to find the interval x belongs to
            i = bisect.bisect_right(xs, val)
            i = i - 1 if i == len(xs) - 1 else i
            c = get_coefs(xs[i], xs[i + 1], 
                          ys[i], ys[i+1],
                          derivs[i], derivs[i+1])
            return (c[0] * val ** 4
                    + c[1] * val ** 3 
                    + c[2] * val ** 2 
                    + c[3] * val)
        else:
            return [get_coefs(xs[i],xs[i+1],
                              ys[i],ys[i+1],
                              derivs[i],derivs[i+1]) for i in range(len(xs)-1)]
    
    #evaluates the interpolating polynomial of xs and ys at x
    def neville_alg(xs, ys, x):
        s = len(xs) #s for size, because it's not the deree
        if len(ys) != s:
            raise Exception("Interpolation unequal number of x and y points")
        polys = [ys[i] for i in range(s)] #cur diagonal of polys
        for i in range(1, s):
            for j in range(s-i):
                polys[j] = ((polys[j]*(x - xs[i+j]) + polys[j+1]*(xs[j]-x)) 
                            / (xs[j] - xs[j+i]))
        return polys[0]
    

#processes for sequence extrapolation (in general, convergence acceration)
class extrapolation:
    #Computes Richardson extrapolation to final value, assuming sequence fed in
    #at index N. The explicit formula used here is nonstandard, but it is able
    #by using Richardson's normal model A(n) = a + b/n + c/n^2 + ... to order R
    #and then simply using Sylvester's formula to find the interpolating poly-
    #nomial of (1/N, A(N)) for our sequence of N and simplifying the result
    def richardson(vals, N):
        #N is the index of the start of vals in the sequence
        R = len(vals) - 1
        r_fac = util.product(lambda x : x+1, range(R)) #r factorial
        tot = 0
        sig = 1 if R % 2 == 0 else -1
        binom_coef = 1
        v = N
        for i in range(R+1):
            elem = vals[i]
            tot += sig * elem * (v ** R) * binom_coef
            #update all the factors
            sig *= -1
            #recursively computes the next binomial coefficient in the row
            binom_coef *= (R - i) / (i + 1)
            v += 1
        return tot / r_fac
    
    #computes the shanks transformation of the sequence. We adopt a convention
    #that the first 2 terms don't change, and the higher order terms are found
    #by shanks-ing each value and it's predecessor
    def shanks_transformation(seq, num_times = 1):
        if num_times == 1:
            #first two terms
            if len(seq) <= 2:
                return seq
            vals = [seq[0], seq[1]]
            #high order terms by shanks
            for i in range(2, len(seq)):
                S0 = seq[i-2] #2 terms back
                S1 = seq[i-1] #1 term bacl
                S2 = seq[i]   #this term
                #result of shanks model
                L = (S0 * S2 - S1 * S1) / (S2 - 2*S1 + S0)
                vals.append(L)
            return vals
        else:
            #recursively compute the n-th order transformation
            return extrapolation.shanks_transformation(
                extrapolation.shanks_transformation(seq, num_times - 1))


#numerical methods for integration
class integration: 
    #left riemann sum in N steps of integral of f on (a, b)
    def riemann_left(f, a, b, N = 10000):
        area = 0
        dx = (b - a) / N
        x = a
        for i in range(N):
            y = f(x)
            dA = y * dx
            area += dA
            x += dx
        return area
    
    #right riemann sum in N steps of integral of f on (a, b)
    def riemann_right(f, a, b, N = 10000):
        area = 0
        dx = (b - a) / N
        x = a + dx
        for i in range(N):
            y = f(x)
            dA = y * dx
            area += dA
            x += dx
        return area
    
    #trapezoidal riemann sum in N steps of integral of f on (a, b)
    def riemann_trap(f, a, b, N = 10000):
        dx = (b - a) / N
        #handle end points first, they only add in once
        tot = .5 * (f(a) + f(b))
        x = a + dx
        for i in range(N-1):
            #all interor points add in 1 time (coef. 1/2 in each neighboring 
            #interval, which inside of the boundaries is 2)
            tot += f(x)
            x += dx
        area = tot * dx
        return area
    
    #monte carlo integration of f on (a, b)
    def monte_carlo(f, a, b, N = 10000):
        #gets a random value from the interval
        get = lambda: a + (b-a) * rand()
        tot = 0
        for i in range(N):
            tot += f(get())
        #expected value of f(g) with g~Unif([a, b])
        avg = tot / N
        #the area from the expecation
        area = (b-a) * avg
        return area

    #simpson's 1/3 rule, or Newton-Cotes quadrature of order 2
    def simpson_1_3(f, a, b, N = 10000):
        #here n is the total number of function evluation (-1, I think)
        #this is true for NC(2), NC(3), but not NC(4) (the same for NC(0) and
        #NC(1)) and then higher order terms where N is used to compute dx and
        #each interval is then subdivided because the composite rules are
        #simpler for low order NC, but for higher orders become much less 
        #intuitive
        
        #handle end points first, since they appear with special coeffs
        tot = (f(a) + f(b))
        mult = 4
        dx = (b - a) / N
        x = a + dx
        
        for i in range(N-1):
            tot += f(x) * mult
            x += dx
            #handle the multiplier in this manner, from composite NC(2) formula
            if mult == 4:
                mult = 2
            else:
                mult = 4
        area = tot * dx / 3
        return area
    
    #simpsons 3/8ths rule, or Newton-Cotes quadrature of order 3
    def simpson_3_8(f, a, b, N = 10000):
        #here n is the total number of function evluation (-1, I think)
        #this is true for NC(2), NC(3), but not NC(4) (the same for NC(0) and
        #NC(1)) and then higher order terms where N is used to compute dx and
        #each interval is then subdivided because the composite rules are
        #simpler for low order NC, but for higher orders become much less 
        #intuitive
        
        #handle end points first, since they appear with special coeffs
        dx = (b - a) / N
        tot = f(a) + f(b)
        x = a + dx
        k = 0
        for i in range(N-1):
            #somewhat convoluted coeff handling, but effectively recursively
            #handle i modulo 3
            if k == 2:
                k = 0
                mult = 2
            else:
                mult = 3
                k += 1
            tot += mult * f(x)
            x += dx
        area = 3 * tot * dx / 8
        return area
        
    #Boole's rule, alternatively called Bode's rule, Newton-Cotes quadrature of
    #order 4
    def boole(f, a, b, N = 2500):
        #here n is the total number of intervals, each using NC(4) (plus or
        #minus an interval. This isn't true for NC(2) and NC(3) (and sort of
        # NC(0) and NC(1), but those are really the same either way. For high
        #order NC, this standard is simpler and more intuitive
        tot = 0
        dx = (b - a) / N
        x = a
        for i in range(N):
            #get necessary values for boole's rule inside the interval
            x0 = x
            x1 = x + .25 * dx
            x2 = x + .5 * dx
            x3 = x + .75 * dx
            x4 = x + dx
            #function values at interval subdivision
            f0 = f(x0)
            f1 = f(x1)
            f2 = f(x2)
            f3 = f(x3)
            f4 = f(x4)
            #this step is actually boole's rule, combined with the 2/45 below
            loc = (7 * f0 + 32 * f1 + 12 * f2 + 32 * f3 + 7 * f4) / 4
            tot += loc
            x += dx
        area = tot * dx * 2 / 45
        return area
    
    #generalized Newton-Cotes quadrature. for order 5 or greater, this is 
    #implemented particularly inefficiently, since it recomputes the Newton-
    #Cotes coefficients each time the function is called, and it computes them
    #by solving a system with a notoriously ill-conditioned Vandermonde matrix.
    #This is mostly ok, because numerical applications of higher order NC are
    #generally rather specialized, so tend to involve just a few calls and 
    #have orders that are not too big. Nonetheless, much to be improved here. 
    def gen_nc(f, a, b, nc_num, N = 10000):
        if nc_num < 0 or nc_num != nc_num // 1:
            raise Exception("Newton Cotes of invalid degree: " + str(nc_num))
        if nc_num == 0:
            return integration.riemann_left(f, a, b, N)
        if nc_num == 1:
            return integration.riemann_trap(f, a, b, N)
        if nc_num == 2:
            return integration.simpson_1_3(f, a, b, 2 * N)
        if nc_num == 3:
            return integration.simpson_3_8(f, a, b, 3 * N)
        if nc_num == 4:
            return integration.boole(f, a, b)
        else:
            #not recommended cuz the alg is bad. Computes coefficients. 
            #uses a classic formula based on a particular vector and a certain
            #vandermonde matrix. 
            d = nc_num
            #vandermonde matrix with multipliers (0, 1, ..., d) in M_(d+1)(R)
            vm = np.zeros((d+1, d+1))
            #initialize the first column to 1
            for i in range(d+1):
                vm[i, 0] = 1
            #use dp to compute the matrix. 
            for i in range(d+1):
                for j in range(1, d+1):
                    vm[i, j] = i * vm[i, j-1]
            #the matrix in the actually formula, isn't vandermonde, it's a 
            #vandermonde matrix transposed. Nonetheless we do this. 
            vm = vm.transpose()
            B = np.zeros(d+1)
            for i in range(d+1):
                B[i] = (nc_num ** (i+1)) / (i+1)
            A = np.linalg.solve(vm, B)
            A /= nc_num
            #A is a vector of our coeficients
            #this is the part where we actually go through the NC for the 
            #particular integral at hand
            area = 0
            x = a
            dx = (b - a) / N
            #for each interval
            for i in range(N):
                #split up in the integral to use the approprtiate NC on it. 
                for j in range(nc_num + 1):
                    xj = x  + j * dx / nc_num
                    yj = f(xj)
                    area += yj * A[j] * dx
                x += dx
            return area
        
    #Implements Romberg's method for integration. Technically, Romberg only 
    #uses the trapezoid rule, but allowing for higher order NC (usually 2 or 3)
    #might be useful, so it is included. This only computes N actual integrals
    #so redundancy in NC of order 5 or higher shouldn't be too bad.
    def romberg(f, a, b, start = 1000, N = 6, nc = 1):
        #N should be about log(# of steps) in this case, so much smaller than 
        #in similar methods
        integral_vals = []
        num_iter = start
        for i in range(N+1):
            integral_vals.append(integration.gen_nc(f, a, b, nc, num_iter))
            num_iter *= 2
        #we just richardson all the way
        return extrapolation.richardson(integral_vals, 1)
    
    #just a special romberg variant that happened to work well for
    #sin(x) on [0, pi]. Useless, just to store it as a possible good choice.
    def opt_rom(f, a, b):
        return integration.romberg(f, a, b, 10000, 8, 4)
    
    #adaptive simpson's method. This definitely fails at getting within err_tol
    #of the answer at all times. I don't know quite, why, but I'm guessing it
    #has to do with machine precision. Never gotten stuck in an infinite loop,
    #though, so I'm not sure. 
    def simpson_adaptive(f, a, b, err_tol = .0000001):
        #return simpson info to use memoization. Most importantly, simple use
        #of simpson's 1/3 rule 1-4-1 coefficients. Todo: add customizability so
        #higher order adaptive NC methods could be used (I mostly just mean 
        #Boole, to high order is obviously a supremely bad idea)
        def simp_mem(f, a, b, fa, fb):
            mid = .5 * (a + b)
            midf = f(mid)
            dx = math.fabs(b-a)
            tot = fa/6 + 2*midf/3 + fb/6
            area = tot * dx
            return mid, midf, area
        
        #recursively compute the result of adaptive simpson's method.
        def rec_adapt_simp(f, a, b, fa, fb, epsilon, area, mid, midf):
            lmid, flmid, left = simp_mem(f, a, mid, fa, midf)
            rmid, frmid, right = simp_mem(f, mid, b, midf, fb)
            diff = left + right - area
            #15 is a magic numebr here that comes from a classical condition
            #for this algorithm
            if diff <= 15 * epsilon:
                return left + right + diff / 15
            else:
                return (rec_adapt_simp(f,a,mid,fa,midf,
                                       epsilon/2,left,lmid,flmid) + 
                        rec_adapt_simp(f,mid,b,midf,fb,
                                       epsilon/2,right,rmid,frmid))         
        #computes the values of f needed to feed the result into a recursive
        #machine
        fa = f(a)
        fb = f(b)
        mid, midf, area = simp_mem(f, a, b, fa, fb)
        return rec_adapt_simp(f, a, b, fa, fb, err_tol, area, mid, midf)
    
    

                        