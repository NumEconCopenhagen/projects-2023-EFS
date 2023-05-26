from types import SimpleNamespace

import numpy as np
from scipy import optimize

class Worker1:

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
        par.tau = np.linspace(1e-8, 1-(1e-8), 500) # labour-income tax rate

        
        par.omega_t = []
        par.el = []
        par.G_vec = []

        sol.L_vec = []
        sol.u_vec = []

     
    def u_func(self,L,g):
        """ calculate utility """
        
        par = self.par
        sol = self.sol

        C = (par.kappa+(1-par.tau_separated)*par.omega*L) # define consumption

        return np.log(C**(par.alpha)*g**(1-par.alpha))-par.nu*L**2/2

    def value_of_choice(self,L,g):
        """ calculate value of choice """

        return -self.u_func(L,g)

    
    def solve(self,do_print=True):
        """ solve model """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        par.omega_t = (1-par.tau)*par.omega # define omega_t
        par.el = ((-par.kappa+ np.sqrt(par.kappa**2+4*(par.alpha/par.nu)*par.omega_t**2))/(2*par.omega_t)) # define q3's L*
        par.G_vec = par.tau * par.omega * par.el * ((1-par.tau) * par.omega) # define vector of G

        guess = 7.0 # initial guess
        bound = (1e-8,24) # bounds for L

        # a. call solver
        for  i, g in enumerate(par.G_vec):
            
            par.tau_separated = par.tau[i] # optimize with one value of tau at a time
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