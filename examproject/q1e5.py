from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Worker2:

    def __init__(self):
        """ setup model parameters """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. baseline parameters 
        par.alpha = 0.5
        par.kappa = 1.0      # free private consumption component
        par.nu = 1/(2*16**2) # disutility of labor scaling factor
        par.omega = 1.0      # real wage
        par.tau = 0.33868 # labour-income tax rate
        par.sigma = 1.001
        par.rho = 1.001
        par.epsilon = 1.0

        par.G_vec = np.linspace(1e-8, 100-(1e-8), 50000) # vector of g's

        sol.L_vec = []
        sol.u_vec = []

     
    def u_func(self,L,g):
        """ calculate utility """
        
        par = self.par
        sol = self.sol

        C = (par.kappa+(1-par.tau)*par.omega*L) # define consumptuon
        
        G = g # par.tau*par.omega*sol_case1 # define government expenditure

        return ((((par.alpha*C**((par.sigma-1)/par.sigma) + (1-par.alpha)*G**((par.sigma-1)/(par.sigma)) )**(par.sigma/(par.sigma-1)) )**(1-par.rho) -1)/(1-par.rho))-par.nu*(L**(1+par.epsilon)/(1+par.rho))

    def value_of_choice(self,L,g):
        """ calculate value of choice """

        return -self.u_func(L,g)

    
    def solve(self,do_print=True):
        """ solve model """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        guess = 7.0 # initial guess
        bound = (1e-8,24) # bounds for L

        # a. call solver
        for  g in par.G_vec:
            
            sol_case1 = optimize.minimize_scalar(
                self.value_of_choice,
                guess,
                method='bounded',
                bounds=(bound),
                args=(g)) 

                # b. append solution
            sol.L_vec.append(sol_case1.x) # append optimal Ls
            sol.u_vec.append(self.u_func(sol_case1.x,g)) # append the utility

        return par.G_vec, sol.L_vec, sol.u_vec, par.tau
    









