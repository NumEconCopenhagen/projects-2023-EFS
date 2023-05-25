from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class Worker:

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
        par.omega_t = (1-par.tau)*par.omega
        par.el = ((-par.kappa+ np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_t**2))/(2*par.omega_t))

        par.G_vec = np.linspace(1.0,2.0,2) 
        sol.L_vec = np.zeros(par.G_vec.size)
        sol.u_vec = np.zeros(par.G_vec.size)

        # f. solution vectors
        sol.L_vec = np.zeros(par.G_vec.size) 
        sol.u_vec = np.zeros(par.G_vec.size) # vector of optimal profit

     
    def u_func(self,L,g):
        """ calculate utility """
        
        par = self.par
        sol = self.sol

        C = (par.kappa+(1-par.tau)*par.omega*L)

        return np.log(C**(par.alpha)*g**(1-par.alpha))-par.nu*L**2/2

    def value_of_choice(self,L,g):
        """ calculate value of choice """

        return -self.u_func(L,g)

    
    def solve(self,do_print=True):
        """ solve model """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        guess = 7.0 # initial guess
        bound = (0.000000000001,24) # bounds for L

        # a. call solver
        for i, g in enumerate(par.G_vec):
            sol_case1 = optimize.minimize_scalar(
                self.value_of_choice,
                guess,
                method='bounded',
                bounds=(0,24),
                args=(g)) # Notice the use of a tuple here

            # b. append solution
            sol.L_vec[i] = sol_case1.x
            sol.u_vec[i] = self.u_func(sol_case1.x,g)

            print(f'For G = {par.G_vec[i]:6.3f}: L = {sol.L_vec[i]:6.3f}, utility = {sol.u_vec[i]:6.3f}, Expected L = {par.el:6.3f} ')

            assert np.isclose(sol.L_vec[i],par.el), 'L and expected L are not close' # check that l and expected l are close
            assert sol.L_vec[i] > 0, 'L is negative' # check that L is positive

        print('\nL and expected L are close and L is positive')