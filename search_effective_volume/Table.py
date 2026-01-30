import matplotlib.pyplot as plt

# Table data
columns = ["R1 (km)", "R2 (km)", r"$V_{\mathrm{eff_new}} / V_{\mathrm{eff}}$", "Percentage improvement"]
data = [
    ["1.00", "1.27", "1.62 – 2.06", "62% – 106%"],
    ["3.00", "3.39", "1.28 – 1.45", "28% – 45%"],
    ["5.00", "5.43", "1.18 – 1.28", "18% – 28%"],
    ["7.00", "7.45", "1.13 – 1.21", "13% – 21%"],
]

# Create figure
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

# Create table
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center",
    cellLoc="center",
)

# Style
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4f8ef7")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#eaf0fb")

# Save as PDF
output_path = "volume_improvement_table_matplotlib.pdf"
plt.savefig(output_path, bbox_inches="tight")
plt.close()

output_path
