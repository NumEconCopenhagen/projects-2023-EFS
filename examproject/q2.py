from types import SimpleNamespace
import numpy as np
from scipy import optimize

class hair_salon():

    def __init__(self,do_print=True):
        """ initialize the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace() # create simplenamespace for parameters
        self.sol = SimpleNamespace() # create simplenamespace for solutions
        self.sim = SimpleNamespace()

        if do_print: print(f'calling .setup()\n')
        self.setup() # calls setup function, defined below

    def setup(self):
        """ setups baseline parameters """

        par = self.par
        sol = self.sol
        sim = self.sim

        # static model parameters
        par.eta = 0.5 # elasticity of demand
        par.w = 1.0 # hairdresser wage
        par.kappa_vec = np.linspace(1.0,2.0,2)

        # static model solution vectors
        sol.l_vec = np.zeros(par.kappa_vec.size) # vector of optimal l
        sol.el_vec = np.zeros(par.kappa_vec.size) # vector of optimal expected l
        sol.profit_vec = np.zeros(par.kappa_vec.size) # vector of optimal profit

        # static model simulation vectors
        sim.kappa = 2.0 # kappa
        sim.l_vec = np.linspace(0.000000000001,5,100) # vector of l
        sim.profit_vec = np.zeros(sim.l_vec.size) # simulated vector of profits

        # dynamic model parameters
        par.rho = 0.9
        par.sigma = 0.1 # std. dev. of random component of demand shocks
        par.iota = 0.01 # fixed adjusment cost for hiring or firing
        par.R = (1+0.01)**(1/12) # monthly discout factor
        par.kappa_init = -1 # initial kappa
        par.l_init = 0.0 # initial l
        par.T = 120 # number of periods
        par.K = 100 # number of random schock series


    def calc_profit(self,l,k):
        """ calculate profit """

        par = self.par
        sol = self.sol

        # a. profit components
        price = k*(l**-par.eta) # implied price
        revenue = price*l
        payroll = par.w*l
        
        return revenue - payroll # profit
    
    
    def value_of_choice(self,l,k):
        """ calculate value of choice """

        return -self.calc_profit(l,k)
    
    
    def expected_optimal_l(self,k):
        """ calculate expected optimal l """
            
        par = self.par
        sol = self.sol
            
        return ((1-par.eta)*k/par.w)**(1/par.eta)
    
    
    def solve(self,do_print=True):
        """ solve model """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        guess = 0.5 # initial guess
        bound = (0.000000000001,100000000000000) # bounds for l

        # b. calculate profit
        for i, k in enumerate(par.kappa_vec):
            opt.l = optimize.minimize_scalar(self.value_of_choice, guess, bounds=bound, method='bounded', args=(k)).x
            
            # append solution vectors
            sol.l_vec[i] = opt.l # store optimal l value in the solution vector
            sol.profit_vec[i] = self.calc_profit(opt.l,k) # store optimal profit in the solution vector
            sol.el_vec[i] = self.expected_optimal_l(k) # store expected optimal l in the solution vector

            print(f'For kappa = {par.kappa_vec[i]:6.3f}: l = {sol.l_vec[i]:6.3f}, profit = {sol.profit_vec[i]:6.3f}, expected l = {sol.el_vec[i]:6.3f}')

            assert np.isclose(sol.l_vec[i],sol.el_vec[i]), 'l and expected l are not close' # check that l and expected l are close
            assert sol.l_vec[i] > 0, 'l is negative' # check that l is positive
        
        print('\nl and expected l are close and l is positive')
            
    
    def plot_profit(self):
        """ plot profit """

        import matplotlib.pyplot as plt

        par = self.par
        sol = self.sol
        sim = self.sim

        for i, k in enumerate(par.kappa_vec):
            for j, l in enumerate(sim.l_vec):
                sim.profit_vec[j] = self.calc_profit(l,k)    
            # a. plot
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot(sim.l_vec,sim.profit_vec)
            ax.plot(sol.l_vec[i],sol.profit_vec[i],'o')
            ax.set_xlabel(r'$\ell_t$')
            ax.set_ylabel(r'$\pi_t$')
            ax.legend([r'$\pi(l)$',r'$\pi(l^*)$'])
            ax.set_title(f'Profit wrt. $\ell_t$, for $\kappa_t={k}$')
            ax.grid(True)

    
    def AR1_demand_shock(self):
        """ AR1 demand shock process """

        return np.exp(par.rho*np.log(k) + par.epsilon)


    def ex_post_profit(self,l,k,t,par):
        """ calculate ex post profit """

        if l[t]==l[t-1]:
            x = 0
        else:
            x = par.iota
        
        return (par.R**-t)*[k*(l**(1-par.eta))-par.w*l-x]
    

    def ex_ante_profit(self,l,k,t,par,K):
        """ calculate ex ante profit """

        for t in reverse(range(par.T)):

            return 

        return
        

    def dynamic_solve(self):
        """ solve dynamic model """

        par = self.par
        sol = self.sol
        sim = self.sim

        # expected value of future profits
        ex_ante = 0.0 
        
        return
    
    def last_period_profit(self,l,k,par):
        """ calculate last period profit """

        return k*(l**(1-par.eta))-par.w*l
    
    def H(self):
        """ calculate H """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. expected profit
        h_plus = 0.0

        for k in len(par.K):
            par.epsilon = np.random.normal(-0.5*par.sigma**2,par.sigma) # draw random part of the demand shock
            par.dyn_kappa = self.AR1_demand_shock(k,par) # calculate kappa
            par.dyn_l = self.expected_optimal_l(par.dyn_kappa) # calculate expected optimal l


