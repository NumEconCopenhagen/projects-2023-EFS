
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
        elif par.sigma == 0:
            H = np.fmin(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
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
        x = np.linspace(0,24,49)
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

    def solve(self,do_print=True):
        """ solve model continously """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        guess = [4.5,4.5,4.5,4.5]
        constraints = ({'type': 'eq', 'fun': lambda x:  24-x[0]-x[1]},{'type': 'eq', 'fun': lambda x:  24-x[2]-x[3]})
        bounds = ((0,24),(0,24),(0,24),(0,24))

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

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages (Question 2/ Question 3)"""
        
        par = self.par
        sol = self.sol
        par.sigma = 1.0
        par.alpha = 0.5
        logHFHM=[]
        logwFwM=[]

        if discrete:
            for i, x in enumerate(list(par.wF_vec)):
                par.wF = x
                optim = self.solve_discrete()
                # Following possible alternative not implemented (I'll delete it if not used: Stefano)
                # sol.LM_vec[i]=opt.LM
                # sol.HM_vec[i]=opt.HM
                # sol.LF_vec[i]=opt.LF
                # sol.HF_vec[i]=opt.HF
                logHFHM.append(np.log(optim.HF/optim.HM))
                logwFwM.append(np.log(x/par.wM))
            par.wF = 1
               
        else:
            for i, x in enumerate(list(par.wF_vec)):
                par.wF = x
                opti = self.solve(do_print=False)
                logHFHM.append(np.log(opti.HF/opti.HM))
                logwFwM.append(np.log(x/par.wM))
            par.wF = 1

        # a. add figure
        fig = plt.figure()

        # b. define plot area
        ax = fig.add_subplot(1,1,1)

        # c. plot type and variables
        ax.plot(logHFHM,logwFwM)

        # d. title, labels
        ax.set_title('Change in '+r'$log\ \frac{H_F}{H_M}$' + ' against ' + r'$log\ \frac{w_F}{w_M}$')
        ax.set_xlabel(r'$log\ \frac{w_F}{w_M}$')
        ax.set_ylabel(r'$log\ \frac{H_F}{H_M}$')

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass

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
            if i > 0:
                text += '\n'
            text += f'{x1:3.2f} ' # left header
            for j, x2 in enumerate(sigma_vec):
                par.alpha = x1
                par.sigma = x2
                text += f'{self.solve_discrete(relH=True):6.3f}'
        
        # d. print
        print(text)