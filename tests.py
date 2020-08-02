import nm
import numpy as np
import math

class util_tests:
    def sum_test():
        triangles = [n * (n+1) // 2 for n in range(1, 11)]
        triangles_summed = [nm.util._sum(lambda x : x, range(i+1)) 
                            for i in range(1, 11)]
        square_sums = [(n * (n+1) * (2*n+1) // 6) for n in range(1, 11)]
        squares_summed = [nm.util._sum(lambda x : x**2, range(i+1)) 
                            for i in range(1, 11)]
        geo_sums = [1 - .5 ** n for n in range(20)]
        geo_summed = [nm.util._sum(lambda x : .5 ** x, range(1, i+1)) 
                            for i in range(20)]
        assert(np.allclose(triangles, triangles_summed))
        assert(np.allclose(square_sums, squares_summed))
        assert(np.allclose(geo_sums, geo_summed))
        
    def prod_test():
        facts = [1, 1, 2, 6, 24, 120]
        facts_prodded = [nm.util.product(lambda i : i, range(1, i+1))
                             for i in range(6)]
        super_facts = [1, 1, 2, 12, 288, 34560]
        super_prodded = [nm.util.product(lambda i : facts[i], range(1, i+1))
                             for i in range(6)]
        geo_prods = [.5 ** ((n * (n+1)) // 2) for n in range(11)]
        geo_prodded = [nm.util.product(lambda i : .5 ** i, range(i+1))
                             for i in range(11)]
        assert(np.allclose(facts, facts_prodded))
        assert(np.allclose(super_facts, super_prodded))
        assert(np.allclose(geo_prods, geo_prodded))
    
    def test_all():
        util_tests.sum_test()
        util_tests.prod_test()

class interpolation_tests:
    def lagrange_test():
        assert(np.allclose(nm.interpolation.lagrange_interpolate([1, 2, 3], 
                                                                 [1, 4, 9]),
                                                                 [0, 0, 1]))
        assert(np.isclose(nm.interpolation.lagrange_interpolate([1, 2, 3], 
                                                                [1, 4, 9], 4),
                                                                16))
    
    def legendre_test():
        data = [[[1, 2, 3], [1, 4, 9], [1/3, 0, 2/3]], 
                [[.5, .3], [2, 1.6], [1, 2]]]
        for xs, ys, ans in data:
            sol = nm.interpolation.legendre_basis_polynomial(xs, ys)
            assert(np.allclose(sol, ans))
            
    def cubic_spline_test():
        data = [[[1, 2, 3], [1, 2, 3], [1, 1, 1], [[0, 0, 1, 0],[0, 0, 1, 0]]],
                [[1, 2, 3], [4, 5, 7], [1, 4, 2], [[3, -12, 16, -3],
                                                   [2, -16, 44, -35]]]]
        for xs, ys, ds, ans in data:
            sol = nm.interpolation.cubic_spline_interpolate(xs, ys, ds)
            for i in range(len(ans)):
                assert(np.allclose(ans[i], sol[i]))
                
    def neville_test():
        assert(np.isclose(nm.interpolation.neville_alg([1, 2, 3], 
                                                       [1, 4, 9], 4),
                                                        16))
        assert(np.isclose(nm.interpolation.neville_alg([1, 2, 3, 4], 
                                                       [1, 8, 27, 64], 5),
                                                        125))
        
    def test_all():
        interpolation_tests.lagrange_test()
        interpolation_tests.legendre_test()
        interpolation_tests.cubic_spline_test()
        interpolation_tests.neville_test()

class extrapolation_tests:
    def richardson_test():
        data = [[[3/2, 1, 5/6, 3/4, 7/10], 1, .5],
                [[2.3449999999999998, 2.30954169797145, 2.2806712962962963, 
                   2.2567137005006828, 2.2365160349854225, 2.2192592592592595, 
                   2.204345703125], 10, 2]]
        for seq, n, a in data:
            r = nm.extrapolation.richardson(seq, n)
            assert(np.isclose(r, a))
        
    def shanks_test():
        data = [[[1, 1.5, 1.75, 1.875, 1.9375], [1, 1.5, 2, 2, 2]],
                [[4.00000000, 2.66666667, 3.46666667, 2.89523810, 
                  3.33968254, 2.97604618, 3.28373848, 3.01707182, 
                  3.25236593, 3.04183962, 3.23231581, 3.05840277, 
                  3.21840277],
                 [4.00000000, 2.66666667, 3.16666667, 3.13333333, 
                  3.14523810, 3.13968254, 3.14271284, 3.14088134, 
                  3.14207182, 3.14125482, 3.14183962, 3.14140672, 
                  3.14173610]]]
        for seq, trans in data:
            shank = nm.extrapolation.shanks_transformation(seq)
            assert(np.allclose(shank, trans))
        
    def test_all():
        extrapolation_tests.richardson_test()
        extrapolation_tests.shanks_test()

class integration_tests:
    def riemann_left_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.riemann_left(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.01, atol=.01))
        
    def riemann_right_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.riemann_right(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.01, atol=.01))
    
    def riemann_trap_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.riemann_trap(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.005, atol=.001))
    
    def monte_carlo_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.monte_carlo(f, a, b, 25000)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard. Also, because of monte-carlo's nature,
            #we check 3/5 times with high values. If This assertion breaks your
            #test, try again a couple of times, because it could just be some
            #especially unlucky random numbers
            successes = 0
            for i in range(5):
                if np.isclose(ans, res, rtol=.1, atol=.1):
                    successes += 1
            assert(successes >= 3)
            
    def simpson_1_3_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.simpson_1_3(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.005, atol=.001))
    
    def simpson_3_8_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.simpson_3_8(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.005, atol=.001))
    
    def boole_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.boole(f, a, b)
            #this assertion has to have special lowered tolerance because
            #integration is rather hard
            assert(np.isclose(ans, res, rtol=.005, atol=.001))
            
    def gen_nc_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        nc_nums = [5, 7, 12, 17]
        for f, a, b, ans in data:
            for nc_num in nc_nums:
                res = nm.integration.gen_nc(f, a, b, nc_num)
                #this assertion has to have special lowered tolerance because
                #integration is rather hard
                assert(np.isclose(ans, res, rtol=.005, atol=.001))        
                
    def romberg_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for ncn in [1, 3, 4]:
            for s in [1000, 2000, 2500]:
                for f, a, b, ans in data:
                        res = nm.integration.romberg(f, a, b, start=s, nc=ncn)
                        #this assertion has to have special lowered tolerance
                        #because integration is rather hard
                        assert(np.isclose(ans, res, rtol=.005, atol=.001))  

    def simp_adaptive_test():
        data = [[math.sin, 0, math.pi, 2], 
                [lambda x : 2 * x, 10, 13, 69],
                [lambda x : 1/x, 1, math.e, 1],
                [lambda x : x ** 3, 0, 1, 1/4],
                [lambda x : math.log(x) * math.log(1-x), 0.000001, .9999999,
                 2 - math.pi ** 2 / 6],
                [lambda x : (math.tan(x/2)/math.sin(x))**2, 0.00001, math.pi/2,
                 2/3]]
        for f, a, b, ans in data:
            res = nm.integration.simpson_adaptive(f, a, b)
            #this assertion has to have special lowered tolerance
            #because integration is rather hard
            assert(np.isclose(ans, res, rtol=.005, atol=.001))
    
    def test_all():
        integration_tests.riemann_left_test()
        integration_tests.riemann_right_test()
        integration_tests.riemann_trap_test()
        integration_tests.monte_carlo_test()
        integration_tests.simpson_1_3_test()
        integration_tests.simpson_3_8_test()
        integration_tests.boole_test()
        integration_tests.gen_nc_test()
        integration_tests.romberg_test()
        integration_tests.simp_adaptive_test()
            
def tot_test():
    util_tests.test_all()
    interpolation_tests.test_all()
    extrapolation_tests.test_all()
    integration_tests.test_all()
    print('All tests passed')
    
tot_test()
        