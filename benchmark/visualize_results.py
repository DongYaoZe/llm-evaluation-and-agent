import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # Support Chinese fonts
plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display

# Load data
csv_path = "results/mmlu_zero_shot.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

# Calculate accuracy by subject
stats = df.groupby("subject").agg(
    total=("is_correct", "count"),
    correct=("is_correct", "sum")
).reset_index()

stats["accuracy"] = (stats["correct"] / stats["total"]) * 100

# Format subject names for display
subject_mapping = {
    "college_biology": "College Biology\n(大学生物学)",
    "college_computer_science": "College Computer Science\n(大学计算机科学)",
    "college_chemistry": "College Chemistry\n(大学化学)"
}
stats["display_name"] = stats["subject"].map(subject_mapping).fillna(stats["subject"])

# Plotting
plt.figure(figsize=(8, 5.5), dpi=300)
colors = ["#1f77b4", "#2ca02c", "#ff7f0e"][:len(stats)]

ax = sns.barplot(
    x="display_name", 
    y="accuracy", 
    data=stats, 
    palette="Blues_d",
    hue="display_name",
    legend=False
)

# Customize the chart style to look premium
plt.title("DeepSeek-V4-Pro on MMLU Benchmark (Zero-Shot)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Subject (学科)", fontsize=12, labelpad=10)
plt.ylabel("Accuracy (准确率 %)", fontsize=12, labelpad=10)
plt.ylim(0, 115)  # Give some headroom for labels

# Add value labels on top of the bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f"{height:.1f}%",
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='center',
        xytext=(0, 10),
        textcoords='offset points',
        fontsize=11,
        fontweight='bold',
        color='#2c3e50'
    )

# Clean up axes
sns.despine(left=True, bottom=True)
plt.tight_layout()

# Save paths
os.makedirs("results", exist_ok=True)
chart_path_results = "results/mmlu_accuracy_chart.png"
plt.savefig(chart_path_results, bbox_inches='tight')
print(f"Chart saved to {chart_path_results}")

# Also copy to Beamer figures folder
ppt_figures_dir = "../../CG组会PPT2/figures"
if os.path.exists(ppt_figures_dir):
    chart_path_ppt = os.path.join(ppt_figures_dir, "mmlu_accuracy_chart.png")
    plt.savefig(chart_path_ppt, bbox_inches='tight')
    print(f"Chart copied to {chart_path_ppt}")
else:
    print(f"Warning: Beamer figures directory {ppt_figures_dir} not found.")

plt.close()
