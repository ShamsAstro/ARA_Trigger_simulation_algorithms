import numpy as np
from scipy.optimize import fsolve

# ------------------------
# User parameters
# ------------------------
kilometers = [1, 3, 5, 7]  # R1 values in km
lam = 1.0                  # km

SNR1 = 2.4563   #old
SNR2 = 2.3488     #new

# ------------------------
# Core equation (solve for R2)
# ------------------------
def equation(R2, R1, lambd, snr1, snr2):
    ratio = snr2 / snr1
    return ratio - (R1 / R2) * np.exp(-(R2 - R1) / lambd)

# ------------------------
# Compute table rows
# ------------------------
rows = []
for R1 in kilometers:
    R2_initial = R1 + 0.01  # initial guess slightly above R1
    R2 = float(fsolve(equation, R2_initial, args=(R1, lam, SNR1, SNR2))[0])

    rr = R2 / R1  # (R2/R1)

    # Eq. (4) bounds
    vmin = rr**2
    vmax = rr**3

    # percentage improvement bounds
    pmin = (vmin - 1.0) * 100.0
    pmax = (vmax - 1.0) * 100.0

    rows.append((R1, R2, vmin, vmax, pmin, pmax))

# ------------------------
# Pretty print like your blue table
# ------------------------
print(f"SNR2/SNR1 = {SNR2/SNR1:.5f}   (SNR1={SNR1}, SNR2={SNR2}, lambda={lam} km)\n")

header = f"{'R1 (km)':>8}  {'R2 (km)':>10}  {'Veff_new/Veff':>18}  {'% improvement':>15}"
print(header)
print("-" * len(header))

for R1, R2, vmin, vmax, pmin, pmax in rows:
    print(f"{R1:8.2f}  {R2:10.2f}  {vmin:7.2f} — {vmax:7.2f}      {pmin:6.0f}% — {pmax:6.0f}%")
