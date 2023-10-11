import unittest
import sys
import numpy as np
import contextlib

sys.path.append('../evolution')
from base import ES

class TestCode(unittest.TestCase):
    
    
    def setUp(self):
        """Setup for test environment for each test"""
        self.bm = ES(algorithm ="MAP",objective= "Hypersphere",dim = 3,population = 15,generations = 40,gamma = .9,number_runs = 4,method = "none",type_covariance = "full")
        self.rl = ES(algorithm = "MAP", objective= "InvertedPendulum-v2", dim = None, population = None, generations = 100,  gamma = .9,number_runs = 4,method =  "none", type_covariance = "full")
        self.bm.Initialize()
        self.bm.current_gen = 0
        self.rl.Initialize()
        self.rl.current_gen = 0
        self.bm.cov = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 7]])
        self.bm.mean = np.array([1, 1, 1])
        self.bm.z = np.array([[2, 2, 1], [2, 3, 1], [1, 4, 7]])
        self.weights = np.array([1/2, 1/6, 1/3])
        self.diag_cov = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 7]])
        self.bm.S = np.linalg.cholesky(self.bm.cov)
        self.bm.beta1 = 0.8
        self.bm.beta2 = 0.999
        self.bm.alpha = .08
        self.bm.cutoff_worse_evaluations = 0
        
    def test_initializer(self):
        """Test if parameters correctly initialized"""
        self.bm.Initialize()
        self.assertIs((self. bm.cov), (self. bm.initial_cov),
           msg= "error in initialization of covariance")
        self.assertIs((self. bm.mean), (self. bm.initial_mean),
           msg= "error in initialization of mean" )
        self.assertFalse(self. bm.linalgerror, msg= "this bool must be False at the beginning of each simulation")
        self.assertEqual(tuple(self. bm.total_time_steps), tuple(np.zeros(self. bm.generations)),
                         msg= "error in initialization of total time steps" )
        self.assertEqual(tuple(self. bm.best_point), tuple(np.zeros(self. bm.generations) ),
                         msg= "error in initialization of median/best point")
        self.assertEqual(tuple(self. bm.first_moment_mean), tuple(np.zeros(self. bm.dim) ),
                         msg= "error in initialization of first moment of mean used in ADAM for SGA/NGA")
        self.assertEqual(tuple(self. bm.second_moment_mean), tuple(np.zeros(self. bm.dim) ),
                         msg= "error in initialization of second moment of mean used in ADAM for SGA/NGA")
        
    def test_vech(self):
        """Test if vech() works correctly"""
        matrix = np.array([[1, 2, 3], [2, 10, 5], [3, 5, 9]])
        result = np.array([1, 2, 10, 3, 5, 9])
        self.assertEqual(tuple(self.bm.vechtorize(matrix)), tuple(result),
                          msg="wrong implemmentation of operator vech()")
    
    def test_math(self):
        """Test if math() works correctly"""
        vector = np.array([1, 2, 10, 3, 5, 9])
        result = np.array([[1., 0., 0.], [2., 10., 0.], [3., 5., 9.]])
        self.assertTrue(np.allclose(self.bm.mathricize(vector), result),
                        msg="wrong implemmentation of inverse operator of vech()")
    
    def test_evaluation_bm(self):
        """Test if the inviduals are correctly evaluated and median stored for Benchmark Optimization"""
        result_true = np.array([0.5, 1/3, 1/6])
        result_false = np.array([3, 2, 1]) 
        self.assertEqual(tuple(self.bm.Expectation(True)), tuple(result_true),
                          msg="wrong function evaluation")
        self.assertEqual(tuple(self.bm.Expectation(False)), tuple(result_false),
                          msg="wrong function evaluation")
        self.assertEqual(self.bm.best_point[self.bm.current_gen], -9.0,
                          msg="median/best point not correctly memorized in vector")
     
        
    def test_stoperror(self):
        """Test if the stopping method for non positive covariance works"""

        with contextlib.redirect_stdout(None):
            self.rl.rl = True
            self.rl.current_gen = 0
            self.rl.min_eig[self.bm.current_gen] = 1e-70
            self.assertTrue(self.bm.StoppingError(),
                msg= "Here the matrix should be marked as non positive definite")

        
    def test_stopcriterium(self):
        """Test if stopping criterium if optimum reached with given tolerance works"""
        self.bm.current_gen = self.bm.generations -1
        with contextlib.redirect_stdout(None):
            self.assertTrue(self.bm.StoppingCriterium(), 
                            msg= "reached last generation, algorithm should stop")
            self.bm.mean = 1e-7*np.zeros(self.bm.dim)
            self.assertTrue(self.bm.StoppingCriterium(), 
                            msg= "reached tolerance, the algorithm should stop")
            self.rl.best_point[self.rl.current_gen] = 1000
            self.assertTrue(self.rl.StoppingCriterium(), 
                            msg= "reached maximum cumulative reward, the algorithm should stop")

        
    def test_cov_mle_update(self):
        """Test if covariance update for MLE algorithm works"""
        
        """full covariance case"""
        self.bm.CovUpdate(self.weights)
        result =np.array([[0.8, 0.85, 0.55], [0.85, 1.35, 1], [0.55, 1, 12.85]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in MLE (full case)")
        
        """diagonal covariance case"""
        self.bm.type_covariance = "diag"
        self.bm.cov = self.diag_cov
        self.bm.CovUpdate(self.weights)
        result =np.array([[0.8, 0, 0], [0, 1.35, 0], [0, 0, 12.85]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in MLE (diag case)")
        
    def test_mean_update(self):
        """Test if mean update for MLE/MAP/SNM algorithms works"""
        self.bm.MeanUpdate(self.weights)
        result =np.array([1.6 , 1.75, 3.25])
        self.assertTrue(np.allclose(self.bm.mean, result), msg="wrong mean update in MLE")      
        
    def test_map_cov_update(self):
        """Test if covariance update for MAP algorithm works"""
        
        """full covariance case"""
        self.bm.ExactUpdate(self.weights)
        result =np.array([[2.24, 2.05  , -0.65  ], [ 2.35  ,  2.5875, -0.3875], [2.65  ,  2.6125 , 9.5875]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in MAP (full case)")
        
        """diagonal covariance case"""
        self.bm.type_covariance = "diag"
        self.bm.cov = self.diag_cov
        self.bm.ExactUpdate(self.weights)
        result =np.array([[5.0084, 0 , 0  ], [ 0 ,   6.238125, 0], [0 ,  0, 26.243125]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in MAP (diag case)")  
    
    def test_snm_cov_update(self):
        """Test if covariance update for SNM algorithm works"""
        
        """full covariance case"""
        self.bm.SNMCovUpdate(self.weights)
        result =np.array([[ 1.2125   ,   0.690625  , -0.996875   ], [  0.690625  ,  1.55568827 ,-5.35595945], [-0.996875  , -5.35595945 ,51.79006588]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in SNM (full case)")
        
        """diagonal covariance case"""
        self.bm.type_covariance = "diag"
        self.bm.cov = self.diag_cov
        self.bm.SNMCovUpdate(self.weights)
        result =np.array([[1.2125 , 0 , 0  ], [ 0 ,   1.88406519, 0], [0 ,  0, 22.652]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in SNM (diag case)")  
        
        
    def test_nga_cov_update(self):
        """Test if covariance update for NGA algorithm works"""
        
        self.bm.algorithm = "NGA"
        
        """full covariance case"""
        self.bm.GACovUpdate(self.weights)
        result =np.array([[1.89475556 ,0.98631111, 0.96035556], [ 0.98631111, 2.85598222, 0.99939911], [0.96035556 ,0.99939911, 7.53103733]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in NGA (full case)")
        
        """diagonal covariance case"""
        self.bm.type_covariance = "diag"
        self.bm.cov = self.diag_cov
        self.bm.GACovUpdate(self.weights)
        result =np.array([[1.89475556, 0 , 0  ], [ 0 ,   2.85512593, 0], [0 ,  0, 7.52965714]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in NGA (diag case)") 
        
        
    def test_sga_cov_update(self):
        """Test if covariance update for SGA algorithm works"""
        
        self.bm.algorithm = "SGA"

        """full covariance case"""
        self.bm.GACovUpdate(self.weights)
        result =np.array([[1.87856806, 0.98451181, 0.96997431], [ 0.98451181 ,2.88764868 ,1.00020993], [0.96997431, 1.00020993 ,7.16856953]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in SGA (full case)")
        
        """diagonal covariance case"""
        self.bm.type_covariance = "diag"
        self.bm.cov = self.diag_cov
        self.bm.GACovUpdate(self.weights)
        result =np.array([[1.89475556, 0 , 0  ], [ 0 ,   2.90301893, 0], [0 ,  0, 7.14935977]])
        self.assertTrue(np.allclose(self.bm.cov, result),
                        msg="wrong covariance update in SGA (diag case)") 
    
    
    def test_nga_mean_update(self):
        """Test if mean update for NGA algorithm works"""
        self.bm.algorithm = "NGA"
        
        """classical update case"""
        self.mean_method_ga = "none"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.1895516409, 1.2369395511, 1.7108186533])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)),
                          msg="wrong mean update in NGA (None method)") 
        
        """ADAM update case"""
        self.mean_method_ga = "adam"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.3791032818, 1.4738791022 , 2.4216373066 ])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)), 
                          msg="wrong mean update in NGA (ADAM method)") 
    
        """Aravinf update case"""
        self.mean_method_ga = "aravind"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.5686549226, 1.7108186533, 3.1324559599 ])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)), 
                          msg="wrong mean update in NGA (Aravind method)") 
        
    def test_sga_mean_update(self):
        """Test if mean update for SGA algorithm works"""
        
        self.bm.algorithm = "SGA"
        
        """classical update case"""
        self.mean_method_ga = "none"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.0228106379, 1.0296538293, 1.0707129776])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)),
                          msg="wrong mean update in SGA (None method)") 
        
        """ADAM update case"""
        self.mean_method_ga = "adam"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.0453612877, 1.058969674 , 1.1416499175 ])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)), 
                          msg="wrong mean update in SGA (ADAM method)") 
        
        """Aravind update case"""
        self.mean_method_ga = "aravind"
        self.bm.GAMeanUpdate(self.weights)
        result =np.array([1.0676600943, 1.0879461268, 1.2128055808 ])
        self.assertEqual(tuple(np.round(self.bm.mean, 10)), tuple(np.round(result,10)), 
                          msg="wrong mean update in SGA (ADAM method)") 
    
    def test_duplication(self):
        """Test if duplication matrix implemented correctly"""
        
        result = np.array([[1., 0., 0., 0., 0., 0.], [0. ,1. ,0. ,0. ,0., 0.], [0., 0. ,0. ,1. ,0. ,0.],
                            [0., 1., 0., 0., 0. ,0.], [0., 0., 1., 0., 0., 0.], [0. ,0. ,0., 0., 1., 0.],
                            [0., 0. ,0., 1., 0. ,0.], [0., 0., 0., 0. ,1., 0.], [0. ,0., 0. ,0. ,0. ,1.]])
        self.assertTrue(np.allclose(self.bm.D(), result),
                        msg="wrong duplication matrix") 
        
    def test_elimination(self):
        """Test if elimination matrix implemented correctly"""
        
        result = np.array([[1., 0., 0., 0., 0., 0.], [0. ,1. ,0. ,0. ,0., 0.], [0., 0. ,0. ,1. ,0. ,0.],
                            [0., 0., 0., 0., 0. ,0.], [0., 0., 1., 0., 0., 0.], [0. ,0. ,0., 0., 1., 0.],
                            [0., 0. ,0., 0., 0. ,0.], [0., 0., 0., 0. ,0., 0.], [0. ,0., 0. ,0. ,0. ,1.]]).T
        self.assertTrue(np.allclose(self.bm.L(), result),
                        msg="wrong elimination matrix") 
    
    
    def test_commutation_Nu(self):
        """Test if commutation matrix implemented correctly"""
        
        result = np.array([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], 
                            [0. , 0.5 ,0.  ,0.5 ,0.,  0.,  0.,  0. , 0. ],
                            [0.,  0. , 0.5 ,0. , 0. , 0. , 0.5, 0.,  0. ],
                            [0. , 0.5, 0.,  0.5, 0. , 0. , 0. , 0. , 0. ],
                            [0. , 0.,  0. , 0. , 1. , 0.,  0. , 0. , 0. ],
                            [0. , 0. , 0. , 0. , 0.,  0.5, 0.,  0.5 ,0. ],
                            [0. , 0. , 0.5 ,0. , 0. , 0. , 0.5, 0. , 0. ],
                            [0. , 0.,  0.  ,0.,  0. , 0.5 ,0. , 0.5, 0. ],
                            [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])

        self.assertTrue(np.allclose(self.bm.Nu(), result),
                        msg="wrong commutation Nu matrix") 
        
    def test_Q(self):
        """Test if Q matrix implemented correctly"""
        
        result = np.array([[2.82842712 ,0.70710678, 0.     ,    0.70710678, 0.    ,     0.      ],
                          [0.    ,     1.41421356 ,1.41421356, 0.      ,   0.70710678 ,0.        ],
                          [0.   ,      0.   ,      3.16227766, 0.       ,  0.31622777 ,0.        ],
                          [0.   ,      0.    ,     0.    ,     1.41421356 ,0.70710678 ,1.41421356],
                          [0.     ,    0.     ,    0.     ,    0.     ,    1.58113883, 0.63245553],
                          [0.     ,    0.      ,   0.    ,     0.        , 0.   ,      5.05964426]])
        self.assertTrue(np.allclose(self.bm.Q(), result),
                        msg="wrong Q matrix")  
        
    def test_dJ_dSIGMA(self):
        """Test if derivative of cholesky decomposition using chain rule  implemented correctly"""
        
        result = np.array([ 0.24088542 ,-0.13216146 , 0.03580729, -0.13216146 , 0.16178385  ,0.00358073,
          0.03580729 , 0.00358073 ,-0.08040365])
        self.assertTrue(np.allclose(self.bm.dJ_dSIGMA(self.weights), result),
                        msg="wrong derivative of cholesky decomposition using chain rule (dJ_dSIGMA)") 
        
    def test_dJ_ds(self):
        """Test if derivative of cholesky decomposition using direct computation implemented correctly"""
        
        result = np.array([-0.54506148 , 0.13994822 ,-0.51387012 , 0.0073657 ,  0.03952847 , 0.40681384])
        self.assertTrue(np.allclose(self.bm.dJ_ds(self.weights), result),
                        msg="wrong derivative of cholesky decomposition using direct computation (dJ_ds)") 
    
    def test_chain_direct(self):
        """Test if derivative using chain rule and direct computation are the same"""
        
        direct = -self.bm.dJ_ds(self.weights)
        chain = self.bm.Q().dot(self.bm.dJ_dSIGMA(self.weights).dot(self.bm.D()))
        self.assertTrue(np.allclose(chain, direct),
                        msg="derivative using chain rule or direct computation must be the same") 
        
    def test_R(self):
        """Test if R matrix implemented correctly"""
        
        result = np.array([[ 0.48177083 ,-0.26432292 , 0.    ,      0.07161458,  0.  ,        0.        ],
                            [-0.26432292 , 0.32356771 , 0.        ,  0.00716146 , 0.     ,     0.        ],
                            [ 0.        ,  0.   ,       0.32356771 , 0.  ,        0.00716146,  0.        ],
                            [ 0.07161458  ,0.00716146 , 0.     ,    -0.16080729 , 0.    ,      0.        ],
                            [ 0.       ,   0.   ,       0.00716146 , 0.   ,      -0.16080729 , 0.        ],
                            [ 0.   ,       0.       ,   0.    ,      0.      ,    0.      ,   -0.16080729]])
        self.assertTrue(np.allclose(self.bm.R(self.bm.dJ_dSIGMA(self.weights).dot(self.bm.D())), result),
                        msg="wrong computation of cross chain term in hessian of SNM (R)")
        
    def test_FIM(self):
        """Test if Fisher Information Matrix w.r.t vechtorized covariance matrix implemented correctly"""
        
        result = np.array([[ 0.1953125 , -0.1171875  , 0.01757812 ,-0.0390625  , 0.01171875  ,0.00195312],
                            [-0.1171875  , 0.2890625 , -0.07617188, -0.0078125  ,-0.01953125,  0.00195312],
                            [ 0.01757812, -0.07617188  ,0.08251953 , 0.00585938, -0.01269531  ,0.00048828],
                            [-0.0390625 , -0.0078125  , 0.00585938 , 0.1015625  ,-0.02734375, -0.00976562],
                            [ 0.01171875, -0.01953125, -0.01269531 ,-0.02734375 , 0.06445312 ,-0.00488281],
                            [ 0.00195312,  0.00195312,  0.00048828 ,-0.00976562 ,-0.00488281 , 0.01220703]])
        self.assertTrue(np.allclose(self.bm.FIM(), result),
                        msg="wrong computation of Fisher Information Matrix w.r.t vechtorized covariance matrix") 
        
    def test_inv_FIM_chol(self):
        """Test if invere Fisher Information Matrix w.r.t vechtorized Cholesky demposition of covariance matrix implemented correctly"""
        
        result = np.array([[1. ,  0.5 , 0.  , 0.5 , 0. ,  0.  ],
                            [0.5 , 2.75,0.   ,0.75 ,0.  , 0.  ],
                            [0. ,  0. ,  1.25 ,0.  , 0.25 ,0.  ],
                            [0.5,  0.75 ,0.  , 6.75, 0.  , 0.  ],
                            [0.  , 0. ,  0.25 ,0.,   6.45, 0.  ],
                            [0.  , 0.  , 0.  , 0.  , 0. ,  3.2 ]])
        self.assertTrue(np.allclose(self.bm.Natty_Inv_FIM(), result),
                        msg="wrong computation of invere Fisher Information Matrix w.r.t vechtorized Cholesky demposition of covariance matrix") 
        
        
if __name__== "__main__":
    unittest.main()
    
    
    
    
    
    
    
    
    
