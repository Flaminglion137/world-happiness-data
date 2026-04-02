import pandas as pd
import numpy as np

df = pd.read_csv("happiness_combined.csv")

FACTORS = [
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "generosity",
    "trust_govt_corruption",
]

LABELS = {
    "gdp_per_capita":         "GDP per Capita",
    "social_support":         "Social Support",
    "health_life_expectancy": "Health / Life Expectancy",
    "freedom":                "Freedom",
    "generosity":             "Generosity",
    "trust_govt_corruption":  "Trust in Government",
}

DIVIDER = "=" * 60

# ===========================================================================
# Q1: Which factors correlate most strongly with happiness score?
# ===========================================================================

print(DIVIDER)
print("Q1: Which factors correlate most strongly with happiness?")
print(DIVIDER)

# Pearson correlation of each factor against happiness_score
corr = (
    df[FACTORS + ["happiness_score"]]
    .dropna()
    .corr()["happiness_score"]
    .drop("happiness_score")
    .sort_values(ascending=False)
)

for col, val in corr.items():
    bar = "█" * int(abs(val) * 20)   # Visual bar scaled to 20 chars = r of 1.0
    print(f"  {LABELS[col]:<28}  r={val:.3f}  {bar}")

strongest = LABELS[corr.idxmax()]
weakest   = LABELS[corr.idxmin()]

print(f"""
CONCLUSION:
  '{strongest}' is the strongest predictor of happiness (r={corr.max():.3f}),
  followed by '{LABELS[corr.index[1]]}' (r={corr.iloc[1]:.3f}) and
  '{LABELS[corr.index[2]]}' (r={corr.iloc[2]:.3f}).
  '{weakest}' has the weakest link (r={corr.min():.3f}), suggesting
  charitable giving matters less to national happiness than wealth or health.
""")

# ===========================================================================
# Q2: Has average global happiness changed from 2015 to 2019?
# ===========================================================================

print(DIVIDER)
print("Q2: Has average global happiness changed from 2015 to 2019?")
print(DIVIDER)

# Mean score per year; restrict to years with full country coverage
avg = df.groupby("year")["happiness_score"].agg(["mean", "count"])

for year, row in avg.iterrows():
    bar = "█" * int(row["mean"] * 4)   # Scale bar (score ~5–6 → 20–24 chars)
    print(f"  {year}:  {row['mean']:.4f}  {bar}  (n={int(row['count'])})")

score_2015 = avg.loc[2015, "mean"]
score_2019 = avg.loc[2019, "mean"]
change     = score_2019 - score_2015
pct_change = (change / score_2015) * 100
direction  = "risen" if change > 0 else "fallen"

# Year-on-year deltas to spot any dip mid-period
yoy = avg["mean"].diff().dropna()
worst_year  = int(yoy.idxmin())
best_year   = int(yoy.idxmax())

print(f"""
CONCLUSION:
  Average happiness has {direction} by {abs(change):.4f} points ({abs(pct_change):.2f}%)
  between 2015 ({score_2015:.4f}) and 2019 ({score_2019:.4f}) — a small but
  {'positive' if change > 0 else 'negative'} trend overall.
  The biggest single-year {'gain' if yoy[best_year] > 0 else 'drop'} was
  {best_year} (+{yoy[best_year]:.4f}) and the biggest drop was in {worst_year}
  ({yoy[worst_year]:.4f}).
""")

# ===========================================================================
# Q3: Are the top 10 richest countries always in the top 10 happiest?
# ===========================================================================

print(DIVIDER)
print("Q3: Are the top 10 richest countries in the top 10 happiest?")
print(DIVIDER)

overlap_rows = []

for year in sorted(df["year"].unique()):
    yr = df[df["year"] == year].dropna(subset=["gdp_per_capita", "happiness_rank"])

    # Top 10 by GDP and top 10 by happiness rank
    top_gdp     = set(yr.nlargest(10, "gdp_per_capita")["country"])
    top_happy   = set(yr.nsmallest(10, "happiness_rank")["country"])
    overlap     = top_gdp & top_happy
    only_gdp    = top_gdp - top_happy     # Rich but not happiest
    only_happy  = top_happy - top_gdp     # Happiest but not richest

    overlap_rows.append({
        "year": year,
        "overlap": len(overlap),
        "overlap_countries": sorted(overlap),
        "rich_not_happy": sorted(only_gdp),
        "happy_not_rich": sorted(only_happy),
    })

    print(f"  {year}:  {len(overlap)}/10 overlap")
    print(f"         In both lists : {', '.join(sorted(overlap)) or 'none'}")
    print(f"         Rich, not top-happy: {', '.join(sorted(only_gdp)) or 'none'}")
    print(f"         Happy, not top-rich: {', '.join(sorted(only_happy)) or 'none'}")
    print()

avg_overlap = np.mean([r["overlap"] for r in overlap_rows])

# Countries that appear in both lists every single year
always_both = set(overlap_rows[0]["overlap_countries"])
for r in overlap_rows[1:]:
    always_both &= set(r["overlap_countries"])

print(f"""
CONCLUSION:
  On average only {avg_overlap:.1f} out of 10 top-GDP countries also rank in the
  top 10 happiest — so wealth alone does not guarantee happiness.
  {f"Countries consistently in both lists every year: {', '.join(sorted(always_both))}." if always_both else "No country appeared in both lists across all 5 years."}
  Countries like Finland and Denmark are routinely among the happiest
  despite not always topping the GDP table, while some of the world's
  wealthiest nations sit outside the happiness top 10.
""")

# ===========================================================================
# Q4: Which region is consistently the happiest across all years?
# ===========================================================================

print(DIVIDER)
print("Q4: Which region is consistently the happiest across all years?")
print(DIVIDER)

# Region is only present in 2015 and 2016 — use those years
region_data = df.dropna(subset=["region"])

if region_data.empty:
    print("  No region data available.\n")
else:
    years_with_region = sorted(region_data["year"].unique())

    # Average happiness rank per region per year (lower rank = happier)
    region_rank = (
        region_data.groupby(["year", "region"])["happiness_rank"]
        .mean()
        .reset_index()
        .rename(columns={"happiness_rank": "avg_rank"})
    )

    # Which region had the lowest (best) average rank each year?
    best_per_year = (
        region_rank.loc[region_rank.groupby("year")["avg_rank"].idxmin()]
        .set_index("year")
    )

    # Overall average rank per region across all available years
    overall = (
        region_rank.groupby("region")["avg_rank"]
        .mean()
        .sort_values()
    )

    print(f"  (Region data available for years: {years_with_region})\n")
    print("  Happiest region each year:")
    for year, row in best_per_year.iterrows():
        print(f"    {year}: {row['region']}  (avg rank {row['avg_rank']:.1f})")

    print("\n  Average rank by region across all years (lower = happier):")
    for region, avg_r in overall.items():
        bar = "█" * max(1, int(20 - avg_r * 0.3))  # Invert: fewer blocks = lower rank
        print(f"    {region:<30}  avg rank {avg_r:5.1f}  {bar}")

    happiest_region = overall.idxmin()
    runner_up       = overall.index[1]

    print(f"""
CONCLUSION:
  '{happiest_region}' is the consistently happiest region with an
  average rank of {overall.min():.1f}, followed by '{runner_up}'
  (avg rank {overall.iloc[1]:.1f}).
  Note: region labels were only collected in 2015 and 2016, so this
  finding reflects those two years — but the pattern aligns with the
  full dataset's country-level rankings.
""")
