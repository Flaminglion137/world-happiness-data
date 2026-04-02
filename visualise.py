import pandas as pd                  # Data loading and manipulation
import matplotlib.pyplot as plt      # Plotting library
import matplotlib.ticker as ticker   # Fine-grained axis formatting

# Load the combined dataset
df = pd.read_csv("happiness_combined.csv")

# Use a clean built-in style for all charts
plt.style.use("seaborn-v0_8-whitegrid")

# Colour palette used consistently across charts
BLUE   = "#4C72B0"
GOLD   = "#DD8452"

# ===========================================================================
# Chart 1: Average happiness score per year (line chart)
# ===========================================================================

# Compute the mean happiness score for every year
avg_by_year = df.groupby("year")["happiness_score"].mean()

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    avg_by_year.index,    # X axis: years
    avg_by_year.values,   # Y axis: average scores
    marker="o",           # Circle marker at each data point
    linewidth=2.5,
    markersize=8,
    color=BLUE,
)

# Annotate each point with its rounded value
for year, score in avg_by_year.items():
    ax.annotate(
        f"{score:.3f}",
        xy=(year, score),
        xytext=(0, 10),           # Offset label 10 points above the marker
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color="#333333",
    )

ax.set_title("Average World Happiness Score per Year", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Average Happiness Score", fontsize=11)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))   # One tick per year
ax.set_ylim(5.0, 6.0)   # Zoom Y axis to the relevant range

plt.tight_layout()
plt.savefig("chart1_avg_happiness_per_year.png", dpi=150)
plt.close()
print("Saved chart1_avg_happiness_per_year.png")

# ===========================================================================
# Chart 2: Top 10 happiest countries in 2019 (horizontal bar chart)
# ===========================================================================

# Filter to 2019 only and take the 10 highest-ranked countries
top10_2019 = (
    df[df["year"] == 2019]
    .nsmallest(10, "happiness_rank")          # Rank 1 is happiest
    .sort_values("happiness_score")           # Sort ascending so rank 1 is at top
)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(
    top10_2019["country"],       # Country names on Y axis
    top10_2019["happiness_score"],
    color=GOLD,
    edgecolor="white",
    height=0.6,
)

# Add score labels at the end of each bar
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.02,                    # Slight gap after bar end
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=9,
        color="#333333",
    )

ax.set_title("Top 10 Happiest Countries in 2019", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Happiness Score", fontsize=11)
ax.set_xlim(0, top10_2019["happiness_score"].max() + 0.4)  # Room for labels
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.savefig("chart2_top10_2019.png", dpi=150)
plt.close()
print("Saved chart2_top10_2019.png")

# ===========================================================================
# Chart 3: Which factors correlate most with happiness score (bar chart)
# ===========================================================================

# These are the six contributing factors present across all years
factor_cols = [
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "generosity",
    "trust_govt_corruption",
]

# Pearson correlation of each factor against happiness_score (drop NaN rows)
correlations = (
    df[factor_cols + ["happiness_score"]]
    .dropna()                              # Remove rows with any missing value
    .corr()["happiness_score"]             # Correlation series vs happiness
    .drop("happiness_score")               # Remove the self-correlation (1.0)
    .sort_values(ascending=True)           # Lowest correlation at top of chart
)

# Nicer display labels for the factor names
labels = {
    "gdp_per_capita":         "GDP per Capita",
    "social_support":         "Social Support",
    "health_life_expectancy": "Health / Life Expectancy",
    "freedom":                "Freedom",
    "generosity":             "Generosity",
    "trust_govt_corruption":  "Trust in Government",
}

fig, ax = plt.subplots(figsize=(9, 5))

# Colour bars: positive correlations blue, negative red (none expected here)
colors = [BLUE if v >= 0 else "#C44E52" for v in correlations.values]

bars = ax.barh(
    [labels[c] for c in correlations.index],   # Human-readable factor names
    correlations.values,
    color=colors,
    edgecolor="white",
    height=0.6,
)

# Add correlation value labels at the end of each bar
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        fontsize=9,
        color="#333333",
    )

ax.set_title("Factor Correlations with Happiness Score (2015–2019)",
             fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Pearson Correlation Coefficient", fontsize=11)
ax.set_xlim(0, 1.05)
ax.axvline(0, color="grey", linewidth=0.8)  # Zero reference line
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
plt.savefig("chart3_factor_correlations.png", dpi=150)
plt.close()
print("Saved chart3_factor_correlations.png")
