import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class consumer:
    
    def __init__(self,**kwargs): # called when created
        
        # a. baseline parameters
        self.alpha = 0.5
        self.kappa = 1.0      # free private consumption component
        self.nu = 1/(2*16**2) # disutility of labor scaling factor
        self.omega = 1.0      # real wage
        self.tau = 0.30 

        self.G = np.linspace(1.0,2.0,2)
        
       self.L = np.linspace(0.0,24.0)
            
        # c. update parameters and settings
        for key, value in kwargs.items():
            setattr(self,key,value) # like self.key = value
        
         # note: "kwargs" is a dictionary with keyword arguments
            
    def __str__(self): # called when printed
        
        lines = f'alpha = {self.alpha:.3f}\n'
        lines += f'price vector = (p1,p2) = ({self.p1:.3f},{self.p2:.3f})\n'
        lines += f'income = I = {self.I:.3f}\n'

        # add lines on solution if it has been calculated
        if not (np.isnan(self.x1) or np.isnan(self.x2)):
            lines += 'solution:\n'
            lines += f' x1 = {self.x1:.2f}\n'
            lines += f' x2 = {self.x2:.2f}\n'
               
        return lines

        # note: \n gives a lineshift

    # utilty function
    def u_func(self,x1,x2):
        return x1**self.alpha*x2**(1-self.alpha)
    
    # solve problem
    def solve(self):
        
        # a. objective function (to minimize) 
        def value_of_choice(x):
            return -self.u_func(x[0],x[1])
        
        # b. constraints
        constraints = ({'type': 'ineq', 'fun': lambda x: self.I-self.p1*x[0]-self.p2*x[1]})
        bounds = ((0,self.I/self.p1),(0,self.I/self.p2))
        
        # c. call solver
        initial_guess = [self.I/self.p1/2,self.I/self.p2/2]
        sol = optimize.minimize(value_of_choice,initial_guess,
                                method='SLSQP',bounds=bounds,constraints=constraints)
        
        # d. save
        self.x1 = sol.x[0]
        self.x2 = sol.x[1]
        self.u = self.u_func(self.x1,self.x2)