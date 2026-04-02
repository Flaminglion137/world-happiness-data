import pandas as pd  # Import pandas for data manipulation

# ---------------------------------------------------------------------------
# Step 1: Load each year's CSV and rename columns to a consistent schema
# ---------------------------------------------------------------------------

# 2015 — has Region, Standard Error, Dystopia Residual
df_2015 = pd.read_csv("2015.csv")
df_2015 = df_2015.rename(columns={
    "Country":                    "country",
    "Region":                     "region",
    "Happiness Rank":             "happiness_rank",
    "Happiness Score":            "happiness_score",
    "Economy (GDP per Capita)":   "gdp_per_capita",
    "Family":                     "social_support",
    "Health (Life Expectancy)":   "health_life_expectancy",
    "Freedom":                    "freedom",
    "Generosity":                 "generosity",
    "Trust (Government Corruption)": "trust_govt_corruption",
    "Dystopia Residual":          "dystopia_residual",
})
df_2015["year"] = 2015  # Tag every row with its source year

# 2016 — has Region, confidence interval columns, Dystopia Residual
df_2016 = pd.read_csv("2016.csv")
df_2016 = df_2016.rename(columns={
    "Country":                    "country",
    "Region":                     "region",
    "Happiness Rank":             "happiness_rank",
    "Happiness Score":            "happiness_score",
    "Economy (GDP per Capita)":   "gdp_per_capita",
    "Family":                     "social_support",
    "Health (Life Expectancy)":   "health_life_expectancy",
    "Freedom":                    "freedom",
    "Generosity":                 "generosity",
    "Trust (Government Corruption)": "trust_govt_corruption",
    "Dystopia Residual":          "dystopia_residual",
})
df_2016["year"] = 2016

# 2017 — dot-notation column names, no Region, has Whisker cols, Dystopia Residual
df_2017 = pd.read_csv("2017.csv")
df_2017 = df_2017.rename(columns={
    "Country":                         "country",
    "Happiness.Rank":                  "happiness_rank",
    "Happiness.Score":                 "happiness_score",
    "Economy..GDP.per.Capita.":        "gdp_per_capita",
    "Family":                          "social_support",
    "Health..Life.Expectancy.":        "health_life_expectancy",
    "Freedom":                         "freedom",
    "Generosity":                      "generosity",
    "Trust..Government.Corruption.":   "trust_govt_corruption",
    "Dystopia.Residual":               "dystopia_residual",
})
df_2017["year"] = 2017

# 2018 — renamed columns, no Region or Dystopia Residual; 1 missing corruption value
df_2018 = pd.read_csv("2018.csv")
df_2018 = df_2018.rename(columns={
    "Country or region":              "country",
    "Overall rank":                   "happiness_rank",
    "Score":                          "happiness_score",
    "GDP per capita":                 "gdp_per_capita",
    "Social support":                 "social_support",
    "Healthy life expectancy":        "health_life_expectancy",
    "Freedom to make life choices":   "freedom",
    "Generosity":                     "generosity",
    "Perceptions of corruption":      "trust_govt_corruption",
})
df_2018["year"] = 2018

# 2019 — same structure as 2018
df_2019 = pd.read_csv("2019.csv")
df_2019 = df_2019.rename(columns={
    "Country or region":              "country",
    "Overall rank":                   "happiness_rank",
    "Score":                          "happiness_score",
    "GDP per capita":                 "gdp_per_capita",
    "Social support":                 "social_support",
    "Healthy life expectancy":        "health_life_expectancy",
    "Freedom to make life choices":   "freedom",
    "Generosity":                     "generosity",
    "Perceptions of corruption":      "trust_govt_corruption",
})
df_2019["year"] = 2019

# ---------------------------------------------------------------------------
# Step 2: Define the final column order — drop year-specific extras
#         (Standard Error, confidence intervals, whiskers) as they don't
#         exist across all years. Columns absent in a year become NaN.
# ---------------------------------------------------------------------------

COLUMNS = [
    "year",
    "country",
    "region",               # NaN for 2017–2019 (not collected)
    "happiness_rank",
    "happiness_score",
    "gdp_per_capita",
    "social_support",
    "health_life_expectancy",
    "freedom",
    "generosity",
    "trust_govt_corruption",
    "dystopia_residual",    # NaN for 2018–2019 (not reported)
]

# ---------------------------------------------------------------------------
# Step 3: Stack all five DataFrames, keeping only the standardised columns
# ---------------------------------------------------------------------------

combined = pd.concat(
    [df_2015, df_2016, df_2017, df_2018, df_2019],
    ignore_index=True,   # Reset the row index so it runs 0…N
)

# Reindex to the target columns; any missing column is filled with NaN
combined = combined.reindex(columns=COLUMNS)

# ---------------------------------------------------------------------------
# Step 4: Save to CSV and print a summary
# ---------------------------------------------------------------------------

combined.to_csv("happiness_combined.csv", index=False)  # No row-number column

print(f"Saved happiness_combined.csv")
print(f"Shape: {combined.shape}")
print(f"\nColumns:\n{list(combined.columns)}")
print(f"\nRows per year:\n{combined['year'].value_counts().sort_index().to_string()}")
print(f"\nMissing values per column:\n{combined.isnull().sum().to_string()}")
print(f"\nFirst 5 rows:")
print(combined.head().to_string(index=False))
