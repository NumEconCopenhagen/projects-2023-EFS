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

        par.G_vec = np.linspace(1.0,2.0,2) 
        sol.L_vec = np.zeros(par.G_vec.size)
        sol.u_vec = np.zeros(par.G_vec.size)

        # f. solution vectors
        sol.L_vec = np.zeros(par.G_vec.size) 
        sol.el_vec = np.zeros(par.G_vec.size) # vector of optimal expected l
        sol.u_vec = np.zeros(par.G_vec.size) # vector of optimal profit

     
    def u_func(self,L,g):
        """ calculate utility """
        
        par = self.par
        sol = self.sol

        return np.log((par.kappa+(1-par.tau)*par.omega*L)**(par.alpha)*g**(1-par.alpha))-par.nu*L**2/2

    def value_of_choice(self,L,g):
        """ calculate value of choice """

        return -u_func(self,L,g)

    def expected_optimal_L(self,g):
        """ calculate expected optimal L """
            
        par = self.par
        sol = self.sol
        
        omega_t = (1-par.tau)*par.omega
            
        return ((-par.kappa+ np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*omega_t))/2*omega_t)
    
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
                args=(g,alpha,kappa,nu,omega,tau)) # Notice the use of a tuple here

            # b. append solution
            sol.L_vec[i] = sol_case1.x
            sol.u_vec[i] = u_func(sol_case1.x,g,alpha,kappa,nu,omega,tau)
            sol.el_vec[i] = self.expected_optimal_L(g)

            print(f'For G = {par.G_vec[i]:6.3f}: L = {L_vec[i]:6.3f}, utility = {u_vec[i]:6.3f}, ')

            assert np.isclose(sol.L_vec[i],sol.el_vec[i]), 'L and expected L are not close' # check that l and expected l are close
            assert sol.l_vec[i] > 0, 'L is negative' # check that L is positive

        return sol.L_vec, sol.u_vec, sol.el_vec