import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import math
import time
import random
import json
from pathlib import Path
from scipy.optimize import curve_fit
from sim_functions import *
from trig_functions import *

# --- Parameters ---
# Baseline (ARA current trigger)
a1, b1 = 4.597144861920596, 2.8810950976647924  # env=10, TOT=0

# Improved trigger
a2, b2 = 5.089001570961028, 2.666990391033165  # env=2, TOT=4

# --- SNR range ---
snr = np.linspace(1, 4.5, 300)

# --- Compute sigmoid curves ---
y1 = sigmoid(snr, a1, b1)
y2 = sigmoid(snr, a2, b2)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(snr, y1, label=f'Baseline (env=10, TOT=0, SNR_50 = {round(b1,2)} )', linewidth=2, color='blue')
plt.plot(snr, y2, label=f'Improved (env=2, TOT=4, SNR_50 = {round(b2,2)} )', linewidth=2, color='orange')

plt.xlabel('SNR', fontsize=12)
plt.ylabel('Pass fraction', fontsize=12)
plt.title('Comparison of Trigger Efficiency before and after TOT', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('sigmoid_comparison.png', dpi=300)
