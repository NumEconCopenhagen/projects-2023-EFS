import sympy as sm
from sympy import *

def log_analytic(beta_val):
    
    """Solves household maximization problem 
    for the optimal rate of savings of young individuals, 
    given logarithmic period utility function, while showing every step of the solution.
    
    Takes no arguments."""

    import sympy as sm
    from IPython.display import display

    # a. initialing math symbols
    n = sm.symbols('n') # population growth
    z = sm.symbols('z') # technological growth
    L0 = sm.symbols('L_0') # starting population
    Lt = sm.symbols('L_t') # young population at time t
    Ltm1 = sm.symbols('L_{t-1}') # old population at time t
    Nt = sm.symbols('N_{t}') # employed young at time t
    Kt = sm.symbols('K_t') # capital rented at time t+1
    Ktm1 = sm.symbols('K_{t-1}') # capital rented at time t
    tauw = sm.symbols('tau_w') # income tax rate
    taur = sm.symbols('tau_r') # interest tax rate
    wt = sm.symbols('w_t') # wage per unit of labor at time 
    rt = sm.symbols('r_t') # economy return on savings at time t
    rtp1 = sm.symbols('r_{t+1}') # economy return on savings between t and t+1
    rtk = sm.symbols('r_t^k') # capital rental rate
    rtb = sm.symbols('r_t^b') # bonds interest rate
    C1t = sm.symbols('c_{1t}') # consumption when young
    C2tp1 = sm.symbols('C_{2t+1}') # consumption when old
    sigma = sm.symbols('sigma') # degree of relative risk aversion
    beta = sm.symbols('beta') # future consumption discount rate
    St = sm.symbols('S_t') # aggregate savings at time t
    st = sm.symbols('s_t') # individual saving rate at time t
    Yt = sm.symbols('Y') # production at time t
    theta = sm.symbols('theta') # degree of substitutability of production factors (if CES production function)
    gamma  = sm.symbols('gamma') # optimal share of factors distribution (if CES production function)
    alpha = sm.symbols('alpha')# output elasticity to factors change (if CD production function)
    A0 = sm.symbols('A_0') # factors productivity level at start
    At = sm.symbols('A_t') # factors productivity level at time t
    Pit = sm.symbols('Pi_t') # profits at time t
    Gt = sm.symbols('G_t') # public consumption at time t
    Bt = sm.symbols('B_t') # value of bonds oustanding at time t
    Btm1 = sm.symbols('B_{t-1}') # value of bonds oustanding at time t-1
    Tt = sm.symbols('T_t') # total tax revenue at time t
    delta = sm.symbols('delta') # capital depreciation rate

    # b. define objective function
    LOGutility = log(C1t)+beta*log(C2tp1)
    print(f'Objective function:')
    display(LOGutility)
    print(f'\n')

    # c. define budget constraints
    savings = sm.Eq(st*(1-tauw)*wt*Nt,St)
    young_consumption = sm.Eq((1-st)*(1-tauw)*wt*Nt,C1t)
    old_consumption = sm.Eq((1+(1-taur)*rtp1)*St,C2tp1)
    print(f'Budget constraints:')
    display(savings,young_consumption,old_consumption)
    print(f'\n')
    
    # d. isolate St from aggregate saving equation:
    St_from_saving = sm.solve(savings, St)

    # e. substitute St inside function for consumption of old agents
    old_consumption_sub = old_consumption.subs(St,St_from_saving[0])
    print(f'Substitution of St inside function for consumption of old agents:')
    display(old_consumption_sub)
    print(f'\n')

    # f. isolate old and young consumption from respective equations:
    C2tp1_from_old = sm.solve(old_consumption_sub, C2tp1)
    C1t_from_young = sm.solve(young_consumption, C1t)

    # g. substitute old and young consumption inside objective function:
    LOGutility_sub = LOGutility.subs(C2tp1,C2tp1_from_old[0])
    LOGutility_sub_sub = LOGutility_sub.subs(C1t,C1t_from_young[0])
    print(f'Substitution of old and young consumption inside objective function:')
    display(LOGutility_sub_sub)
    print(f'\n')

    # h. first order derivative of the modified objective function:
    foc_household = sm.diff(LOGutility_sub_sub, st)
    print(f'FOC of objective modified function wrt st:')
    display(foc_household)
    print(f'\n')

    # h. putting FOC equal to zero and solving for value of st:
    sol_household = sm.solve(sm.Eq(foc_household,0), st)
    print(f'Value of st that solves FOC=0:')
    display(sol_household[0])
    print(f'\n')

    # i. characterizes the solution by selecting a  
    sol_func = sm.lambdify(args=(beta),expr=sol_household[0])
    opt_savings = sol_func(beta_val)
    print(f'Optimal value of st for beta = {beta_val}:')
    display(opt_savings)