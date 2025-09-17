import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input file
IN_JSON = Path("threshold_scan_rates_120sec.json")

def main():
    # Load results
    with open(IN_JSON) as f:
        results = json.load(f)

    # Extract data
    thresholds = [r["threshold"] for r in results]
    trigger_rates = [r["trigger_rate"] for r in results]
    tot_values = [r["tot_samples"] for r in results]  # list of lists

    # --- 1) Trigger rate vs threshold ---
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, np.log(trigger_rates), marker="o", linestyle="-", label="Trigger rate")
    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("Trigger rate (triggers / events) log scale")
    plt.title("Trigger Rate vs Threshold (pure noise)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trigger_rate_vs_threshold_1h_log.png")


    # --- 2) TOT vs threshold ---
    # Flatten: for each threshold, we have 15 TOT values
    # We can plot them as scatter or boxplot
    plt.figure(figsize=(8,6))
    for thr, tots in zip(thresholds, tot_values):
        plt.scatter([thr]*len(tots), tots, alpha=0.6, label=None, color="tab:blue")

    plt.xlabel("Threshold (ADC²)")
    plt.ylabel("TOT (samples)")
    plt.title("Time Over Threshold vs Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tot_vs_threshold_1h.png")


if __name__ == "__main__":
    main()
