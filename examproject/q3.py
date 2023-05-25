import numpy as np
from scipy.optimize import minimize
import math



def griewank(x):
    """Define Griewank function"""
    return griewank_(x[0],x[1])
    
def griewank_(x1,x2):
    A = x1**2/4000 + x2**2/4000
    B = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(1))
    return A-B+1

def refined_global_optimizer(bounds, tolerance, warmup_iterations, max_iterations):
    
    x_best = None  # Best solution found so far
    f_best = np.inf  # Best function value found so far
    initial_guesses = []  # Store effective initial guesses

    for iteration in range(max_iterations):
        # Draw random uniformly within chosen bounds
        x_initial = np.random.uniform(bounds[0], bounds[1], len(bounds))

        if iteration < warmup_iterations:
            # Run BFGS optimizer with initial guess
            x_initial_best = minimize(griewank, x_initial, method='BFGS', tol=tolerance)

        else:
            capital_x_initial = 0.50 * 2/(1+math.exp((iteration - warmup_iterations)/100))
            x_initial_zero = capital_x_initial*x_initial + (1-capital_x_initial)*x_best
            x_initial_best = minimize(griewank, x_initial_zero, method='BFGS', tol=tolerance)

        # Check if the optimizer successfully converged
        if x_initial_best.success:
            x_current = x_initial_best.x
            f_current = x_initial_best.fun

            # Update the best solution if a new best is found
            if f_current < f_best or iteration == 0:
                x_best = x_current
                f_best = f_current
                if iteration >= warmup_iterations:
                    best_iteration = iteration
                    print(f'{iteration:4d}: x0 = ({x_initial_zero[0]:7.2f},{x_initial_zero[1]:7.2f})',end='')
                    print(f' -> converged at ({x_best[0]:7.2f},{x_best[1]:7.2f}) with f = {f_best:12.8f}')
        # Check if warm-up iterations are completed
        #if iteration >= warmup_iterations:
            # Check if the difference between the current best and initial guess is greater than tolerance
            #if np.linalg.norm(x_best - x_initial) > tolerance:
               # x_initial = x_best
            #if f_best < tolerance:
                #return x_best

        # Check if warm-up iterations are completed
        if iteration >= warmup_iterations:
            # Store the effective initial guess
            initial_guesses.append(x_initial_zero)

    print(f'Best iteration counting warm up{best_iteration:4d}\n',end='')
    print(f'Best iteration not counting warm up{best_iteration - warmup_iterations:4d}',end='')

    return x_best, initial_guesses, best_iteration

def average_list(list):
    return sum(list)/len(list)

