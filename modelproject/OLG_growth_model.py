from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class OLGModelClass():

    def __init__(self,do_print=True):
        """ initialize the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace() # create simplenamespace for parameters
        self.sim = SimpleNamespace() # create simplenamespace for simulation variables

        if do_print: print('calling .setup()')
        self.setup() # calls setup function, defined below

        if do_print: print('calling .allocate()')
        self.allocate() # calls allocation function, defined below
    
    def setup(self):
        """ setups baseline parameters """

        par = self.par

        # a. household
        par.sigma = 1.0 # CRRA coefficient
        par.beta = 1.0/(1.0+0.4) # discount factor

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.5 # capital weight
        par.theta = 0.0 # substitution parameter
        par.delta = 0.5 # depreciation rate

        # c. government
        par.tau_w = 0.0 # labor income tax
        par.tau_r = 0.0 # capital income tax
        par.bal_budget = True


        # d. stocks
        par.K_lag_ini = 1.0 # initial capital stock
        par.B_lag_ini = 0.0 # initial government debt

        # e. timeframe
        par.simT = 50 # length of simulation

        # f. population
        par.L_lag_ini = 1.0 # initial population
        par.n = 0.1 # population growth rate

        # g. steady-state levels conditional of type of production function
        if par.production_function == 'cobb-douglas':
            par.k_ss = ((1-par.alpha)/((1+par.n)*(1+1.0/par.beta)))**(1/(1-par.alpha)) # steady-state level of capital for Log utility and Cobb Douglas production


    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par 
        sim = self.sim

        # a. create a list of variables (at period t, otherwise)
        household = ['C1','C2'] # (C1): consumption of young, (C2): consumption of old
        firm = ['Y','K','K_lag','k','k_lag'] # (Y): aggregate production, (K): aggregate capital, (K_lag): aggregate capital at t-1, (k): per-worker capital
        prices = ['w','rk','rb','r','rt'] # (w): wage, (rk): capital rental rate, (rb): bonds interest rate, (r): after-depreciation return , (rt): after-tax return
        government = ['G','T','B','balanced_budget','B_lag'] # (G): dovernment spending, (T): government income, (B): bond outstanding value, (Balanc.): boolean for respecting budget constraint, (B_lag): bond oustanding at t-1
        population = ['L','L_lag'] # (L): population, (L_lag): population at time t

        # b. creates an empty array for each variable, as an instance of the model attribute sim 
        allvarnames = household + firm + prices + government + population # unique list of variables
        for varname in allvarnames: # loops trough all variables in the unique list
            sim.__dict__[varname] = np.nan*np.ones(par.simT) # creates the empty array


    def simulate(self,do_print=True):
        """ simulates model """

        t0 = time.time() # starts time counting

        par = self.par
        sim = self.sim
        
        # a. sets initial values for stocks and population
        sim.K_lag[0] = par.K_lag_ini
        sim.k_lag[0] = par.K_lag_ini/par.L_lag_ini
        sim.B_lag[0] = par.B_lag_ini
        sim.L_lag[0] = par.L_lag_ini
        # print(f'\nK_lag[0] = {sim.K_lag[0]},k_lag[0] = {sim.k_lag[0]}, L_lag[0] = {sim.L_lag[0]}')

        # b. iterates over the number of periods of the simulation
        for t in range(par.simT):

            # i. simulates variable values before s is decided
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue  # stops if we are at the second-last period         

            # ii. find bracket to use within which searching for the optimal saving rate
            s_min,s_max = find_s_bracket(par,sim,t)

            # iii. find optimal value of saving rate
            obj = lambda s: calc_euler_error(s,par,sim,t=t) # objective function
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect') # optimization wrt s
            s = result.root # optimal value of s

            # iv. simulates variable values after s is decided
            simulate_after_s(par,sim,t,s)
            # print(f'\nt = {0}, K_lag[{t}] = {sim.K_lag[t]}, k_lag[{t}] = {sim.k_lag[t]}, L_lag[{t}] = {sim.L_lag[t]}')

        if do_print: print(f'\nsimulation done in {time.time()-t0:.2f} secs') # prints time elapsed from start of simulation

        print(f'\noptimal saving rate in period 49 = {s:3f}') # prints result in last period

def find_s_bracket(par,sim,t,maxiter=10000,do_print=False):
    """ find bracket for s to search in """

    # a. defines maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. finds value and sign of euler error when maximum saving rate is used
    value = calc_euler_error(s_max,par,sim,t) # stores value of euler error with s = s_max
    sign_max = np.sign(value) # stores sign of euler error with s = s_max
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min # copy upper bound of bracket
    upper = s_max # copy lower bound of bracket

    it = 0 # starting iteration counter
    while it < maxiter:
                
        # i. calculates midpoint between bracket and uses it to calculate euler error
        s = (lower+upper)/2 
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. checks conditions
        valid = not np.isnan(value) #checks result is not a nan
        correct_sign = np.sign(value)*sign_max < 0 # check new error value has opposite sign respect to s_max option
        
        # iii. 
        if valid and correct_sign: 
            s_min = s # the midpoint of the previous bracket becomes the min value of the new bracket
            s_max = upper # upper remains the max value of bracket
            if do_print: # prints bracket if activated
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # value is nan: too low s: need to increase lower bound
            lower = s
        else: # sign is not changed respect to previous s_max too high: too high s -> decrease upper bound
            upper = s

        # iv. increment iteration step
        it += 1

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # simulate for the next period

    # c. define Euler equation sides
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS # return the difference between the RH and LH sides (Euler error)

def simulate_before_s(par,sim,t):
    """ simulate forward """

    if t > 0: # for every period different from the starting point, we update lag variables.
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]
        sim.L_lag[t] = sim.L_lag[0]*(1+par.n)**t
        sim.k_lag[t] = sim.K_lag[t]/sim.L_lag[t]

    # a. production and factor prices
    if par.production_function == 'ces':

        # i. production
        sim.Y[t] = (par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(sim.L_lag[t])**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(sim.L_lag[t])**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. production
        sim.Y[t] = (sim.K_lag[t]**par.alpha) * ((sim.L_lag[t])**(1-par.alpha))

        # ii. factor prices
        sim.rk[t] = par.alpha * (sim.K_lag[t]**(par.alpha-1)) * ((sim.L_lag[t])**(1-par.alpha))
        sim.w[t] = (1-par.alpha) * (sim.K_lag[t]**(par.alpha)) * ((sim.L_lag[t])**(-par.alpha))

    else:

        raise NotImplementedError('unknown type of production function')

    # b. no-arbitrage and after-tax return
    sim.r[t] = sim.rk[t]-par.delta # after-depreciation return
    sim.rb[t] = sim.r[t] # same return on bonds
    sim.rt[t] = (1-par.tau_r)*sim.r[t] # after-tax return

    # c. consumption
    sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+sim.B_lag[t]) #

    # d. government
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]+sim.B_lag[t]) + par.tau_w*sim.w[t]*sim.L_lag[t]
    
    if par.bal_budget == True: # if government run's a balanced budget
        sim.balanced_budget[:] = True 

    if sim.balanced_budget[t]: # imposes balance budget condition if required
        sim.G[t] = sim.T[t] - sim.r[t]*sim.B_lag[t]

    sim.B[t] = (1+sim.r[t])*sim.B_lag[t] - sim.T[t] + sim.G[t] #

    # #e. population in period t
    sim.L[t] = sim.L_lag[t]

def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young in period t
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*sim.L_lag[t]*(1.0-s)

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t] # define investment of peridod t
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I # define aggregate capital of peridod t (ready to use for period t+1)

def capital_accumulation_plot(par,sim):
        """ plots the graph for capital accumulation against the define steady-state level of capital 
        
        Attributes:
        - par = model parameter attributes
        - sim = model simulation atributes
        """

        fig = plt.figure(figsize=(6,6/1.5)) # decide figure size
        ax = fig.add_subplot(1,1,1) # decide figure plots number and position
        ax.plot(sim.k_lag,label=r'$k_{t}$') # plot the capital accumulation curve with label
        ax.axhline(par.k_ss,ls='--',color='black',label='analytical steady state') #plots csteady-state line with label
        ax.legend(frameon=True) # add legend box
        fig.tight_layout() # adjust the padding between and around subplots.

def population_plot(par,sim):
        """ plots the graph for population growth 
        
        Attributes:
        - par = model parameter attributes
        - sim = model simulation atributes
        """

        fig = plt.figure(figsize=(6,6/1.5)) # decide figure size
        ax = fig.add_subplot(1,1,1) # decide figure plots number and position
        ax.plot(sim.L_lag,label=r'$L_{t}$') # plot the population curve with label
        ax.legend(frameon=True) # add legend box
        fig.tight_layout() # adjust the padding between and around subplots.