import math, time

import scipy
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import pybobyqa

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class KalmanFilter(object):

    def __init__(self, dim_x, dim_y, dim_u=0):

        self.dim_x = dim_x # dimension of state variables
        self.dim_y = dim_y # dimension of measurement variables
        self.dim_u = dim_u # dimension of control variables (defaulted at 0)

        # initial state estimate, x ~ N(x0, P0)
        # use diffuse prior unless explicitly provided
        self.kappa = 1e+12 # parameter for diffuse prior
        self.x0 = np.zeros((dim_x, 1))
        self.P0 = self.kappa * np.eye(dim_x)

        # placeholder for observation
        self.Y = []

        # favorite inverse/determinant operation
        self.inv = scipy.linalg.inv
        self.det = scipy.linalg.det

        # clear up model parameters
        self._reset_model()

    def _reset_model(self):

        dim_x = self.dim_x
        dim_y = self.dim_y
        dim_u = self.dim_u

        # initialize state estimate, x ~ N(x,P)
        self.x = self.x0
        self.P = self.P0

        # placeholder for state transition equations: x' = Ax + v + nu where v = Bu and nu ~ N(0,V)
        self.A = np.eye(dim_x)
        self.B = np.zeros((dim_x, dim_u))
        self.u = np.zeros((dim_u, 1))
        self.v = self.B @ self.u
        self.V = np.eye(dim_x)

        # placeholder for measurement equations: y' = Cx' + w + omega, omega ~ N(0,W)
        self.C = np.zeros((dim_y, dim_x))
        self.w = np.zeros((dim_y, 1))
        self.W = np.eye(dim_y)

        # placeholder for prior on next state x'|y ~ N(x_prior, P_prior)
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # placeholder for forecast on measurement y'|y ~ N(y,Q)
        self.y = np.zeros((dim_y, 1))
        self.Q = np.eye(dim_y)
        self.Qinv = self.inv(self.Q)

        # placeholder for Kalman gain K and prediction error q
        self.K = np.zeros((dim_x, dim_y))
        self.q = np.zeros((dim_y, 1))

    def predict(self, A, B, u, V, C, w, W):
        """
        Calculate prior x'|y and forecast y'|y
        """
        #
        # compute state prior x_{t}|Y_{t-1} ~ N(x_{t|t-1}, P_{t|t-1})
        #
        # - self.x: posterior mean from previous period
        # - self.P: posterior covariance from previous period

        # prior mean: x_{t|t-1} = A_{t}x_{t-1|t-1} + v_{t}
        self.x_prior = A @ self.x + B @ u

        # prior covariance: P_{t|t-1} = AP_{t-1|t-1}A.T + V_{t}
        self.P_prior = A @ self.P @ A.T + V

        #
        # compute forecast y_{t}|Y_{t-1} ~ N(y_{t|t-1}, Q_{t|t-1})
        #

        # mean: y_{t|t-1} = C_{t}x_{t|t-1} + w_{t}
        self.y = C @ self.x_prior + w

        # covariance: Q_{t|t-1} = C_{t}P_{t|t-1}C_{t}.T + W_{t}
        self.Q = C @ self.P_prior @ C.T + W
        self.Qinv = self.inv(self.Q)

        # Kalman gain: K_{t} = P_{t|t-1} * C_{t}.T * Q_{t|t-1}^{-1}
        self.K = self.P_prior @ C.T @ self.Qinv

    def update(self, y):
        """
        Calculate posterior x'|y'
        - y in the input is observed value for y'
        """

        # Kalman gain calculated in the self.predict() step
        K = self.K

        #
        # compute posterior x_{t}|Y_{t} ~ N(x_{t|t}, P_{t|t})
        #

        # prediction error
        self.q = y - self.y

        # posterior mean: x_{t|t} = x_{t|t-1} + K_{t}q_{t}
        self.x = self.x_prior + K @ self.q

        # posterior covariance: P_{t|t} = P_{t|t-1} - KQ_{t|t-1}K.T
        self.P = self.P_prior - K @ self.Q @ K.T

    def Pt(self):
        """
        Calculate Prob(y_{t}|Y_{t-1})
        """
        #
        val = np.exp(-1/2 * (self.q.T @ self.Qinv @ self.q)[0][0])
        val = val / (2*np.pi)**(self.dim_y/2)
        val = val / (abs(self.det(self.Q)))**(1/2)
        return val

    def log_Pt(self, checkError=True):
        """
        Calculate ln(Prob(y_{t}|Y_{t-1}))
        """
        val = np.log(self.Pt())

        if checkError and (np.isinf(val) or np.isnan(val)):
            raise ValueError("log_Pt takes an invalid value (inf or nan).")

        return val

    def set_observation(self, Y):
        self.Y = Y

    def log_likelihood(self, A, B, u, V, C, w, W, verbose=False, checkError=True):

        # reset model (clear x, y, A, v, V, C, w, W)
        # use x0, P0 stored at self.x0, self.P0
        self._reset_model()

        Y = self.Y

        val = 0
        for y in Y:

            # set self.y, self.x_prior, self.Q, self.Qinv, and self.K
            self.predict(A, B, u, V, C, w, W)

            # set self.x, self.P, self.y, self.q
            self.update(y)

            # compute Pt and add log(Pt) to output value
            val += self.log_Pt(checkError)

        if verbose:
            print(val)

        return val


class Model(object):

    def __init__(self):

        # kalman filter instance
        dim_x, dim_y  = 3, 2
        self.kf = KalmanFilter(dim_x, dim_y)

        # estimation results
        self.mle = {}

    def build_matrices(self, parameters):

        gamma, chi1, chi2, kappa1, kappa2, epsilon, sigma1, sigma2, sigma3, Fbar = parameters

        # -- continuous model ---
        # xdot = A * x + v + disturbance
        # y = C * x + w + error

        # -- discretized model ---
        # x' = Ad * x + vd + nu where vd = Bd * u
        # y' = Cd * x' + wd + omega

        # A matrix
        A = np.array(
            [[-gamma,                               0.0,                     0.0],
             [1/chi1, -(kappa1 + epsilon + kappa2)/chi1, (kappa2 + epsilon)/chi1],
             [   0.0,                       kappa2/chi2,            -kappa2/chi2]])
        m = A.shape[0]

        # B matrix
        B = np.zeros((m,1))
        B[0] = gamma

        # V matrix
        V = np.zeros((m, m))
        V[0,0] = sigma1**2
        V[1,1] = (sigma2/chi1)**2
        V[2,2] = (sigma3/chi2)**2

        # discretize A, B, V
        Ad = scipy.linalg.expm(A)
        Bd = np.linalg.solve(A, (Ad - np.eye(m))) @ B
        D = np.block(
            [[-A,               V  ],
             [np.zeros((m, m)), A.T]])
        F = scipy.linalg.expm(D)
        Vd = F[m:2*m, m:2*m].T @ F[0:m, m:2*m]

        # x0
        ud0 = np.array([[0]]) # no external forcing up until t=0
        x0 = np.linalg.solve((np.eye(m) - Ad), Bd @ ud0)

        # P0
        vecP0 = np.linalg.solve(np.eye(m * m) - np.kron(Ad, Ad), Vd.ravel(order='F'))
        P0 = vecP0.reshape(m, m, order='F')

        # Cd
        Cd = np.array(
            [[0.0,               1.0,     0.0],
             [1.0, -kappa1 - epsilon, epsilon]])

        # Wd, almost zero matrix
        Wd = np.eye(Cd.shape[0]) * 1e-15

        # vd = Bd * u and wd
        ud = np.array([[Fbar]])
        wd = np.zeros((Cd.shape[0],1))

        return Ad, Bd, ud, Vd, Cd, wd, Wd, x0, P0

    def generate_sample(self, n, seed=None):

        if seed is not None:
            np.random.seed(seed=seed)

        # MIROC6 parameter values estimated based on CMIP6 abrupt-4x
        # take these values as `true' parameter values and generate a sample
        gamma = 1.99369386052016
        chi1 = 5.16173008064773
        chi2 = 356.372094143013
        kappa1 = 1.46040898686255
        kappa2 = 1.05771442660202
        epsilon = 0.351749086954843
        sigma1 = 0.753477129440748
        sigma2 = 1.07870003355975
        sigma3 = 0.576467962298279
        Fbar = 9.61971990592549

        self.parameters_true = gamma, chi1, chi2, kappa1, kappa2, epsilon, sigma1, sigma2, sigma3, Fbar
        Ad, Bd, ud, Vd, Cd, wd, Wd, x0, P0 = self.build_matrices(self.parameters_true)

        # draw initial state x0 from N(x0, P0)
        P0_sqrt = np.linalg.cholesky(P0)
        x0 = x0 + P0_sqrt @ np.random.randn(P0_sqrt.shape[-1]).reshape(-1,1)

        # draw state disturbance nu from N(0, Vd)
        Vd_sqrt = np.linalg.cholesky(Vd)
        Nu = [Vd_sqrt @ np.random.randn(Vd_sqrt.shape[-1]).reshape(-1,1) for _ in range(n)]

        # draw measurement error omega from N(0, Wd)
        Wd_sqrt = np.linalg.cholesky(Wd)
        Omega = [Wd_sqrt @ np.random.randn(Wd_sqrt.shape[-1]).reshape(-1,1) for _ in range(n)]

        # generate a sample
        X, Y = [], []
        x_prev = x0
        for nu, omega in zip(Nu, Omega):

            # state transition
            x = Ad @ x_prev + Bd @ ud + nu
            X.append(x)

            # measurement
            y = Cd @ x + wd + omega
            Y.append(y)

            # update previous state
            x_prev = x

        self.kf.set_observation(Y)

    def objfun(self, input_values, log_input=True, verbose=False, checkError=True):
        '''
        objective function to minimize (negative of log likelihood)
        '''
        if log_input:
            # if input is log(parameters)
            parameters = np.exp(input_values)
        else:
            parameters = input_values
        Ad, Bd, ud, Vd, Cd, wd, Wd, x0, P0 = self.build_matrices(parameters)

        # initial state distribution x ~ N(x0, P0)
        self.kf.x0, self.kf.P0 = x0, P0

        return -self.kf.log_likelihood(Ad, Bd, ud, Vd, Cd, wd, Wd, verbose, checkError)

    def estimate(self, verbose=True):

        tol = 1e-5
        maxiter = 10000
        #methods = ['BFGS', 'Powell', 'COBYQA', 'BOBYQA', 'SLSQP', 'Nelder-Mead']
        #methods = ['BFGS', 'SLSQP', 'Nelder-Mead']
        methods = ['BFGS', 'Nelder-Mead']
        #methods = ['BFGS']

        self.methods = methods
        self.num_attempts = 2

        # initial guess
        parameters = self.parameters_true
        #parameters = None # begin with random initial guess

        # initialize best fvalue and best_parameters
        best_fvalue = np.inf
        best_parameters = parameters

        results = {} # store result for each method
        for attempt in range(self.num_attempts):

            print(f"### Attempt {attempt+1}/{self.num_attempts} ###\n")

            # initial guess
            fvalue = np.inf
            print('Searching for a good initial guess in a neighborhood of the current best estimate...')
            scale = attempt/2 if attempt else 1.0e-3
            for _ in range(500):
                log_parameters_candidate = np.log(best_parameters) + np.random.randn(10) * scale
                try:
                    fvalue0 = self.objfun(log_parameters_candidate, verbose=False)
                except Exception as e:
                    #print(e)
                    continue
                if fvalue0 < fvalue:
                    parameters = np.exp(log_parameters_candidate)
                    fvalue = fvalue0
                    print(f" - fvalue: {fvalue}")
            print()

            bounds = [(None, None) for _ in parameters]

            # find the minimizing point using multiple methods
            for method in self.methods:
                success = False
                try:
                    print(f"Solving for MLE with {method}")
                    if method == 'BOBYQA':
                        lower = []
                        upper = []
                        bounds_bobyqa = [(np.log(1.0e-4), np.log(1.0e+4)) for _ in parameters]
                        for bound in bounds_bobyqa:
                            lower.append(bound[0])
                            upper.append(bound[1])
                        seek_global_minimum = False # takes longer time if True
                        start_time = time.time()
                        res = pybobyqa.solve(objfun=self.objfun, x0=np.log(parameters), maxfun=maxiter, bounds=(lower, upper), scaling_within_bounds=True, seek_global_minimum=seek_global_minimum)
                        elapsed_time = time.time() - start_time
                        success = res.flag == res.EXIT_SUCCESS
                        fvalue = res.f
                        message = res.msg
                        num_iter = res.nf
                        parameters = np.exp(res.x)
                    else:
                        start_time = time.time()
                        options = {
                            'maxiter': maxiter,
                        }
                        if method == 'Nelder-Mead':
                            options['adaptive'] = True
                        if method == 'BFGS':
                            options['gtol'] = 1.0e-3
                        if method == 'COBYQA':
                            options['scale'] = True
                        res = scipy.optimize.minimize(fun=self.objfun, x0=np.log(parameters), method=method, bounds=bounds, tol=tol, options=options)
                        elapsed_time = time.time() - start_time
                        success = res.success
                        fvalue = res.fun
                        message = res.message
                        num_iter = res.nit
                        parameters = np.exp(res.x)
                        if method == 'BFGS':
                            hess_inv = res.hess_inv
                            # get the standard errors of parameters from standard errors of log(parameters)
                            jacobian = np.diag(parameters) # Jacobian of the exp(parameters) evaluated at the MLE value
                            covariance = jacobian @ hess_inv @ jacobian.T # covariance of MLE parameters
                            std_errs = np.sqrt(np.diag(covariance))
                        else:
                            checkError = False # otherwise numerically computing Hessian would raise an error
                            hessian = nd.Hessian(lambda log_parameters: self.objfun(log_parameters, checkError=checkError))
                            hess_inv = scipy.linalg.inv(hessian(res.x))
                            # get the standard errors of parameters from standard errors of log(parameters)
                            jacobian = np.diag(parameters) # Jacobian of the exp(parameters) evaluated at the MLE value
                            covariance = jacobian @ hess_inv @ jacobian.T # covariance of MLE parameters
                            std_errs = np.sqrt(np.diag(covariance))

                        confidence_intvls = 1.959*std_errs # 95% interval

                except Exception as e:
                    print(f"===> Error in {method}: {e}\n")
                    continue

                status = 'Success' if success else 'Failure'
                print(f"- {status} in {elapsed_time:.3f} seconds: {message} ({num_iter} iterations)")

                if success:
                    print(f"- fvalue = {fvalue}")
                    #print(f"- at following parameters:")
                    ##gamma, chi1, chi2, kappa1, kappa2, epsilon, sigma1, sigma2, sigma3, Fbar = parameters
                    #for parameter, parameter_true in zip(parameters, parameters_true):
                    #    print(f"  {parameter:.4f} ({parameter_true:.4f})")

                    # update the best estimate for a given method
                    if (method not in results) or (fvalue < results[method]['fvalue']):
                        results[method] = {
                            'attempt': attempt,
                            'elapsed_time': elapsed_time,
                            'parameters': parameters,
                            'fvalue': fvalue,
                            'message': message,
                            'status': status,
                            'confidence_intvls': confidence_intvls,
                            'res': res,
                        }
                        print(f'===> Best estimate for {method} updated')

                    # update the best estimate among all methods
                    if fvalue < best_fvalue:
                        print('===> Best estimate updated')
                        best_method = method
                        best_parameters = parameters # update min_parameters
                        best_fvalue = fvalue # update min_fvalue
                print()

        print('\n=== Summary ===\n')
        print(f" - sample size n: {n}")
        print(f" - seed: {seed}")
        print()
        for method in results:
            fvalue = results[method]['fvalue']
            status = results[method]['status']
            attempt = results[method]['attempt']
            elapsed_time = results[method]['elapsed_time']
            message = results[method]['message']
            parameters = list(results[method]['parameters'])
            if method == best_method:
                print(f"*** {method} (Best method)")
                self.mle[seed] = parameters
            else:
                print(f"*** {method}")
            print(f" - fvalue: {fvalue} (attempt {attempt+1})")
            print(f" - status: {status} in {elapsed_time} seconds")
            print(f" - message: {message}")
            print(f" - estimated parameters (vs true parameter values):")
            for parameter, parameter_true, ce in zip(parameters, self.parameters_true, confidence_intvls):
                print(f"  {parameter:.4f} +-{ce:.4f} ({parameter_true:.4f})")

            print()


if __name__ == "__main__":

    m = Model()

    n = 250 # number of measurement
    seed = 0

    m.generate_sample(n=n, seed=seed)
    m.estimate()
    #parameters = m.mle[seed]
