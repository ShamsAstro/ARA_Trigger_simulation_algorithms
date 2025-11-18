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


#NEW CSW FFT
#Sigmoid parameters: a = 10.718979305764995, b = 1.7438539445873382 (50% efficiency SNR), CSW threshold = 51.4
#OLD/Current ARA trigger Algorithm
#Sigmoid parameters: a = 7.7603860683716706, b = 2.4598937591794923 (50% efficiency SNR), THRESHOLD_V = 71727



# --- Parameters ---
# Baseline (ARA current trigger)
a1, b1 =7.7603860683716706, 2.4598937591794923  
# Improved trigger
a2, b2 = 10.718979305764995, 1.7438539445873382

# --- SNR range ---
snr = np.linspace(1, 4, 300)

# --- Compute sigmoid curves ---
y1 = sigmoid(snr, a1, b1)
y2 = sigmoid(snr, a2, b2)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(snr, y1, label=f'Baseline (Current ARA Trigger - env=10 - SNR_50 = {round(b1,2)} )', linewidth=2, color='blue')
plt.plot(snr, y2, label=f'Improved (CSW_FFT Trigger - N=10 - SNR_50 = {round(b2,2)} )', linewidth=2, color='red')

plt.axhline(y=0.5, color='green', linestyle='--')
plt.axvline(x=b1, color='blue', linestyle='--')
plt.axvline(x=b2, color='red', linestyle='--')


plt.xlabel('SNR', fontsize=12)
plt.ylabel('Pass fraction', fontsize=12)
plt.title('Contrasting the efficiency of CSW and current ARA trigger algorithms', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
#make legend smaller
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('sigmoid_comparison_CSWFFT_normal.png', dpi=300)
plt.show()