
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HSMC:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1.0:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0.0:
            H = np.fmin(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False,relH=False,relL=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(1e-8,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        # f.return HF/HM or LF/LM
        if relH:
            return opt.HF/opt.HM
        if relL:
            return opt.LF/opt.LM

        return opt

    def value_of_choice(self,x):
        return -self.calc_utility(x[0],x[1],x[2],x[3])

    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        #i. set up parameters (initial guess, constraints & bounds) for optimization
        guess = [4.5,4.5,4.5,4.5]
        constraints = ({'type': 'ineq', 'fun': lambda x:  24-x[0]-x[1]},{'type': 'ineq', 'fun': lambda x:  24-x[2]-x[3]})
        bounds = ((0,24),(0,24),(0,24),(0,24))

        #ii. optimize (minimize) uisng SLSQP method
        j = optimize.minimize(
            self.value_of_choice, guess,
            method='SLSQP', constraints=constraints, bounds=bounds)
        
        opt.LM = j.x[0]
        opt.HM = j.x[1]
        opt.LF = j.x[2]
        opt.HF = j.x[3]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt   

    def solve_wF_vec(self, discrete=False, Print=True):
        """ solve model for vector of female wages (Question 2/ Question 3)"""
        
        # a. class parameters
        par = self.par
        sol = self.sol

        # b. initialize vector of results
        logHFHM=[]
        logwFwM=[]

        if Print:
            print(f'For sigma = {par.sigma:6.3f}, alpha = {par.alpha:6.3f}:')

        # c. loop over wF vector 
        for i, x in enumerate(par.wF_vec):
            with np.errstate(all='ignore'):
                par.wF = x
            
                # c.i. use discrete or continuous solver
                if discrete:
                    optim = self.solve_discrete()
                else:
                    optim = self.solve()
            
                # c.ii append class solution vectors
                # j = np.where(par.wF_vec==1)[0][0]
                sol.HM_vec[i]=optim.HM
                sol.HF_vec[i]=optim.HF
                sol.LM_vec[i]=optim.LM
                sol.LF_vec[i]=optim.LF
            
                # c.iii print results
                if Print:
                    print(f'For wF = {x:6.3f} -> optimal HM = {optim.HM:6.3f}; optimal HF = {optim.HF:6.3f} -> HF/HM = {optim.HF/optim.HM:6.3f}, log HF/HM = {np.log(optim.HF/optim.HM):6.3f}')
            
                # c.iV append vectors of results
                logHFHM.append(np.log(optim.HF/optim.HM))
                logwFwM.append(np.log(x/par.wM))

        return logwFwM, logHFHM
        


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
        """ estimate alpha and sigma """
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
        obj = lambda y: objective(y)
        guess = [0.5, 0.5]
        bounds = [(0.0, 1.)] * 2

        # ii. optimize (minimize) using Nelder-Mead method. Find minimum value with variable alpha and 
        #sigma
        result = optimize.minimize(obj,
                            guess,
                            method='Nelder-Mead',
                            bounds=bounds)
        
        #iii. print the solutions of the optimization for alpha & sigma, and the minimum value obtained
        print("alpha = ", result['x'][1])
        print("sigma = ", result['x'][0])
        print("The minimum value obtained is ", result.fun)
        
        
    def modification(self, alph):
        """ Estimate sigma given an alpha"""
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
        print("The minimum value obtained is ", result.fun)


    def tableHFHM(self,alpha_vec,sigma_vec):
        """ HF/HM table for sigma and alpha val (Question 1) """

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