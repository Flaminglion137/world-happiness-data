import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Load the combined dataset
df = pd.read_csv("happiness_combined.csv")

# Factor columns shared across all years (no NaN issues)
FACTORS = [
    "happiness_score",
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "generosity",
    "trust_govt_corruption",
]

# Human-readable labels for those columns
LABELS = {
    "happiness_score":        "Happiness Score",
    "gdp_per_capita":         "GDP per Capita",
    "social_support":         "Social Support",
    "health_life_expectancy": "Health / Life Expectancy",
    "freedom":                "Freedom",
    "generosity":             "Generosity",
    "trust_govt_corruption":  "Trust in Government",
}

plt.style.use("seaborn-v0_8-whitegrid")

# ===========================================================================
# 1. Summary statistics — mean, min, max for all numeric columns
# ===========================================================================

numeric_cols = df.select_dtypes(include="number").columns  # All numeric columns
stats = df[numeric_cols].agg(["mean", "min", "max"])       # Three rows of stats

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
# Round to 3 dp and transpose so columns are rows (easier to read)
print(stats.round(3).T.to_string())
print()

# ===========================================================================
# Chart 1: GDP per Capita vs Happiness Score (scatter)
# ===========================================================================

# Drop rows with missing values in either column
scatter_data = df[["gdp_per_capita", "happiness_score", "year"]].dropna()

fig, ax = plt.subplots(figsize=(9, 6))

# Colour each year differently so multi-year overlap is visible
years = sorted(scatter_data["year"].unique())
palette = plt.cm.tab10.colors  # Up to 10 distinct colours

for i, year in enumerate(years):
    subset = scatter_data[scatter_data["year"] == year]
    ax.scatter(
        subset["gdp_per_capita"],
        subset["happiness_score"],
        label=str(year),
        color=palette[i],
        alpha=0.65,        # Slight transparency to show overlapping points
        s=45,              # Marker size
        edgecolors="white",
        linewidths=0.4,
    )

# Add an overall linear trend line across all years
import numpy as np
x = scatter_data["gdp_per_capita"].values
y = scatter_data["happiness_score"].values
m, b = np.polyfit(x, y, 1)                      # Fit degree-1 polynomial (line)
x_line = np.linspace(x.min(), x.max(), 200)
ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5,
        linestyle="--", label="Trend line")

ax.set_title("GDP per Capita vs Happiness Score (2015–2019)",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("GDP per Capita", fontsize=11)
ax.set_ylabel("Happiness Score", fontsize=11)
ax.legend(title="Year", fontsize=9, title_fontsize=9)

plt.tight_layout()
plt.savefig("chart4_gdp_vs_happiness.png", dpi=150)
plt.close()
print("Saved chart4_gdp_vs_happiness.png")

# ===========================================================================
# Chart 2: Freedom vs Happiness Score (scatter)
# ===========================================================================

scatter_data2 = df[["freedom", "happiness_score", "year"]].dropna()

fig, ax = plt.subplots(figsize=(9, 6))

for i, year in enumerate(years):
    subset = scatter_data2[scatter_data2["year"] == year]
    ax.scatter(
        subset["freedom"],
        subset["happiness_score"],
        label=str(year),
        color=palette[i],
        alpha=0.65,
        s=45,
        edgecolors="white",
        linewidths=0.4,
    )

# Trend line for freedom vs happiness
x2 = scatter_data2["freedom"].values
y2 = scatter_data2["happiness_score"].values
m2, b2 = np.polyfit(x2, y2, 1)
x_line2 = np.linspace(x2.min(), x2.max(), 200)
ax.plot(x_line2, m2 * x_line2 + b2, color="black", linewidth=1.5,
        linestyle="--", label="Trend line")

ax.set_title("Freedom vs Happiness Score (2015–2019)",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Freedom to Make Life Choices", fontsize=11)
ax.set_ylabel("Happiness Score", fontsize=11)
ax.legend(title="Year", fontsize=9, title_fontsize=9)

plt.tight_layout()
plt.savefig("chart5_freedom_vs_happiness.png", dpi=150)
plt.close()
print("Saved chart5_freedom_vs_happiness.png")

# ===========================================================================
# Chart 3: Correlation heatmap for all factors
# ===========================================================================

# Compute pairwise Pearson correlations; drop rows with any NaN first
corr_data = df[FACTORS].dropna()
corr_matrix = corr_data.corr()

# Rename matrix axes to human-readable labels for display
corr_matrix.index   = [LABELS[c] for c in corr_matrix.index]
corr_matrix.columns = [LABELS[c] for c in corr_matrix.columns]

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    corr_matrix,
    annot=True,          # Print the correlation value inside each cell
    fmt=".2f",           # Round to 2 decimal places
    cmap="coolwarm",     # Blue = negative, Red = positive correlation
    center=0,            # Centre the colour scale at zero
    vmin=-1, vmax=1,     # Full correlation range
    linewidths=0.5,      # Thin grid lines between cells
    square=True,         # Keep cells square for readability
    ax=ax,
    annot_kws={"size": 9},
)

ax.set_title("Factor Correlation Heatmap (2015–2019)",
             fontsize=14, fontweight="bold", pad=14)

# Rotate axis labels so they don't overlap
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("chart6_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved chart6_correlation_heatmap.png")
