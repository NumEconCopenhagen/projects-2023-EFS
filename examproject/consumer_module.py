import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class consumer:
    
    def __init__(self,**kwargs): # called when created
        
        # a. baseline parameters
        self.alpha = 0.5
  
        self.alpha = 0.5
        self.kappa = 1.0      # free private consumption component
        self.nu = 1/(2*16**2) # disutility of labor scaling factor
        self.omega = 1.0      # real wage
        self.tau = 0.30     # labour-income tax rate
        
        self.G = np.linspace(1.0,2.0,1)
        self.L = np.nan
        
            
        # c. update parameters and settings
        for key, value in kwargs.items():
            setattr(self,key,value) # like self.key = value
        
         # note: "kwargs" is a dictionary with keyword arguments
            
    def __str__(self): # called when printed
        
        lines = f'solution?\n'
   

        # add lines on solution if it has been calculated
        if not (np.isnan(self.x1) or np.isnan(self.x2)):
            lines += 'solution:\n'
            lines += f' L = {self.L:.2f}\n'
            lines += f'G = {self.G:.3f}\n'
               
        return lines

    # utilty function
    def u_func(self,L):

        # a. (consumption) constraint 
        C = self.kappa + (1 - self.tau) * self.omega * self.L

        # c. total consumption utility
        utility = np.ln(C**(self.alpha)*self.G**(1-self.alpha))

        return utility
    
    
    # solve problem
    def solve(self):
        
        # a. objective function (to minimize) 
        def value_of_choice(L):
            return -self.u_func(L[0])
        
        # b. constraints
        constraints = ({'type': 'ineq', 'fun': lambda L:  self.C-self.kappa+(1-self.tau)*self.omega*L[0]})
        bounds = (0,24)
        
        # c. call solver
        initial_guess = [5,5]
        sol = optimize.minimize(value_of_choice,initial_guess,
                                method='SLSQP',bounds=bounds,constraints=constraints)
        
        # d. save
        self.L = sol.x[0]
        self.u = self.u_func(self.L)
 
    # find indifference curves
    def find_indifference_curves(self):
        
        # allocate memory
        self.x1_vecs = []
        self.x2_vecs = []
        self.us = []
        
        for fac in [0.5,1,1.5]:
            
            # fac = 1 -> indifference curve through optimum
            # fac < 1 -> ... below optimum
            # fac > 1 -> ... above optimum
                
            # a. utility in (fac*x1,fac*x2)
            u = self.u_func(fac*self.x1,fac*self.x2)
            
            # b. allocate numpy arrays
            x1_vec = np.empty(self.N)
            x2_vec = np.linspace(1e-8,self.x2_max,self.N)

            # c. loop through x2 and find x1
            for i,x2 in enumerate(x2_vec):

                # local function given value of u and x2
                def objective(x1):
                    return self.u_func(x1,x2)-u
            
                sol = optimize.root(objective, 0)
                x1_vec[i] = sol.x[0]
            
            # d. save
            self.x1_vecs.append(x1_vec)
            self.x2_vecs.append(x2_vec)
            self.us.append(u)
    
    # plot budgetset
    def plot_budgetset(self,ax):
        
        x = [0,0,self.I/self.p1] # x-cordinates in triangle
        y = [0,self.I/self.p2,0] # y-cordinates in triangle
        
        # fill triangle
        ax.fill(x,y,color="firebrick",lw=2,alpha=0.5) # alpha controls transparance
        
    # plot solution
    def plot_solution(self,ax):
        
        ax.plot(self.x1,self.x2,'ro') # a black dot
        ax.text(self.x1*1.03,self.x2*1.03,f'$u^{{max}} = {self.u:.2f}$')
        
    # plot indifference curve
    def plot_indifference_curves(self,ax):
        
        for x1_vec,x2_vec,u in zip(self.x1_vecs,self.x2_vecs,self.us):
            ax.plot(x1_vec,x2_vec,label=f'$u = {u:.2f}$')
    
    # details of the plot (label,limits,grid,legend)
    def plot_details(self,ax):

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
                
        ax.set_xlim([0,self.x2_max])
        ax.set_ylim([0,self.x2_max])

        ax.grid(ls='--',lw=1)
        ax.legend(loc='upper right')


    def plot_everything(self):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)

        self.plot_indifference_curves(ax)
        self.plot_budgetset(ax)
        self.plot_solution(ax)
        self.plot_details(ax)