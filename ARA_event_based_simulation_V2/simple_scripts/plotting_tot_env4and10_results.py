import matplotlib.pyplot as plt

# Data
TOT_4 = list(range(0, 11))
TOT_10 = list([0,2,3,4,5,6,7,8,9,10])
SNR_env10 = [2.46, 2.43, 2.40, 2.41, 2.42, 2.40, 2.43, 2.46, 2.45, 2.42]
SNR_env4 =  [2.47, 2.46, 2.48, 2.46, 2.54, 2.75, 2.69, 2.45, 2.38, 2.38, 2.44]

# Plot
plt.figure(figsize=(8,5))
plt.plot(TOT_10, SNR_env10, marker='o', label='Envelope = 10')
plt.plot(TOT_4, SNR_env4,  marker='s', label='Envelope = 4')

# Labels and title
plt.xlabel("TOT threshold (≥ samples)")
plt.ylabel("50% Efficiency SNR")
plt.title("Comparing TOT Algorithm with ARA-like Pulse at env = 4 and 10")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Show
plt.tight_layout()
plt.savefig("TOT_efficiency_comparison_env4and10.png")
#plt.show()
