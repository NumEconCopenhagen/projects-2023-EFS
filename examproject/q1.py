from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Household:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. baseline parameters 
        par.alpha = 0.5
        par.kappa = 1.0      # free private consumption component
        par.nu = 1/(2*16**2) # disutility of labor scaling factor
        par.omega = 1.0      # real wage
        par.tau = 0.30     # labour-income tax rate

        par.G = np.linspace(1.0,2.0,1) # come back maybe 2

        # f. solution vectors
        sol.L_vec = np.zeros(par.G.size) 
     

    def calc_utility(self,L):
        """ calculate utility """

        par = self.par
        sol = self.sol


        # a. (consumption) constraint 
        C = par.kappa + (1 - par.tau) * par.omega * L

        # b. wage ?
        #omega_tilde = (1 - par.tau) * par.omega
        
        # c. total consumption utility
        utility = np.fmax(np.ln((C,1e-8)**(par.alpha)*(G,1e-8)**(1-par.alpha))) # why ,1e-8 ?

        # d. disutility of work
        disutility = par.nu*L**2/2
        
        return utility - disutility


    def value_of_choice(self,x):
        """ defines objective function and feeds positional arguments"""
        return -self.calc_utility(L[0])

    def solve(self,do_print=True):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        #i. set up parameters (initial guess, constraints & bounds) for optimization
        guess = [4.5]
        constraints = ({'type': 'ineq', 'fun': lambda x:  24-x[0]-x[1]}) # come back
        bounds = (0,24)

        #ii. optimize (minimize) uisng SLSQP method
        j = optimize.minimize(
            self.value_of_choice, guess,
            method='SLSQP', constraints=constraints, bounds=bounds, tol = 0.000000000001) # added a low tolerance level to better optimize results
        
        opt.L = j.x[0]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt   
    
    
    def solve2(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        #i. set up parameters (initial guess, constraints & bounds) for optimization
        guess = [4.5]
        constraints = ({'type': 'ineq', 'fun': lambda x:  24-x[0]})
        bounds = (0,24)

        #ii. optimize (minimize) uisng SLSQP method
        j = optimize.minimize(
            self.value_of_choice, guess,
            method='SLSQP', constraints=constraints, bounds=bounds, tol = 0.000000000001) # added a low tolerance level to better optimize results
        
        opt.LM = j.x[0]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt   

    def solve_L_vec(self, discrete=False, Print=True):
        """ solve model for vector of female wages (Question 2/ Question 3)"""
        
        # a. class parameters
        par = self.par
        sol = self.sol

        # b. initialize vector of results
        optL=[]

        if Print:
            print(f'For sigma = {par.sigma:6.3f}, alpha = {par.alpha:6.3f}:')

        # c. loop over wF vector 
        for i, x in enumerate(par.wF_vec):
            with np.errstate(all='ignore'):
                par.wF = x
            
                # c.i. use continuous solver
                optim = self.solve()
            
                # c.ii append class solution vectors
                sol.L_vec[i]=optim.L
            
                # c.iii print results
                if Print:
                    print(f'For L = {x:6.3f}')
            
                # c.iV append vectors of results
                optL.append(np.optim.L)

        return optL
        
        
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        # i. use the log of the previously computed vectors to do the regression and obtain
        #beta zero and beta one
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec) # we want to change these
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    

    def estimate(self):
        """ estimate alpha and sigma (Question 4) """
        par = self.par
        sol = self.sol

        # i. objective function (to minimize)
        def objective(y):
            par.alpha = y[1] #both alpha and sigma are variable
            par.sigma = y[0] 
            self.solve_wF_vec(Print=False)
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        #i. set up parameters (function, initial guess & bounds) for the optimization
        guess = [0.5, 0.5]
        bounds = [(0.0, 1.)] * 2

        # ii. optimize (minimize) using Nelder-Mead method. Find minimum value with variable alpha and 
        #sigma
        result = optimize.minimize(objective,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds)
        
        #iii. print the solutions of the optimization for alpha & sigma, and the minimum value obtained
        print("alpha = ", result['x'][1])
        print("sigma = ", result['x'][0])
        print("The minimum value obtained is", result.fun)
        
        
    def modification(self, alph):
        """ Estimates sigma given an alpha (Question 5) """
        par = self.par
        sol = self.sol

        # i. objective function (to minimize)
        def objective(y):
            par.alpha = alph #chosen alpha
            par.sigma = y #variable
            self.solve_wF_vec(Print=False)
            self.run_regression()
            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        #i. set up parameters (function, initial guess & bounds) for the optimization        
        obj = lambda y: objective(y)
        guess = [0.5]
        bounds = [(0.0, 100.)]

        # ii. optimizer (minimize) using Nelder-Mead method. In this case only minimum value and 
        #sigma are to be found since alpha is being provided
        result = optimize.minimize(obj,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds)
        
        #iii. print the solutions for the optimization given for sigma and the minimum value obtained
        print("sigma = ", result.x)
        print("The minimum value obtained is", result.fun)


    def tableHFHM(self,alpha_vec,sigma_vec):
        """ HF/HM table for values of sigma and alpha (Question 1) """

        par = self.par
        sol = self.sol

        # a. empty text
        text = ''
    
        # b. top header
        text += 'a|s'
        text += f'{"":4s}'
        for j, x2 in enumerate(sigma_vec):
            text += f'{x2:3.2f}'
            text += f'{"":2s}'
        text += '\n'
    
        # c. body
        for i,x1 in enumerate(alpha_vec):
            with np.errstate(all='ignore'):
                if i > 0:   
                    text += '\n'
                text += f'{x1:3.2f} ' # left header
                for j, x2 in enumerate(sigma_vec):
                    par.alpha = x1
                    par.sigma = x2
                    text += f'{self.solve_discrete(relH=True):6.3f}'
        
        # d. print
        print(text)