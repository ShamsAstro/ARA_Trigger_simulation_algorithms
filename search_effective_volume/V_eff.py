import numpy as np
from scipy.optimize import fsolve

kilometers=[1,3,5,7]

def equation(R2, R1, lambd):
    ratio = SNR2 / SNR1
    return ratio - (R1 / R2) * np.exp(-(R2 - R1) / lambd)


for r in kilometers:
    # Given values
    R1 = r        # km
    lam = 1.0       # km
    SNR1 = 2.46
    SNR2 = 1.47
    
    #SNR1 = 2.88
    #SNR2 = 2.67
    R2_initial = R1 + 0.001  # Initial guess for R2


    R_2_solution = fsolve(equation, R2_initial, args=(R1, lam))[0]
    print(f"For R1 = {R1} km:")
    print(f"R2 = {R_2_solution:.4f} km  ")
