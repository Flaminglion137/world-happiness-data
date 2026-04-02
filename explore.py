import pandas as pd  # Import pandas for loading and inspecting CSV data

# Define all 5 dataset files with their corresponding years
files = {
    2015: "2015.csv",
    2016: "2016.csv",
    2017: "2017.csv",
    2018: "2018.csv",
    2019: "2019.csv",
}

# Loop through each year and file
for year, filename in files.items():
    print("=" * 60)
    print(f"  YEAR: {year}  |  FILE: {filename}")
    print("=" * 60)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Show the number of rows and columns
    print(f"\nShape (rows, columns): {df.shape}")

    # Show all column names
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    # Show each column's data type
    print("\nData types:")
    print(df.dtypes.to_string())

    # Show how many values are missing in each column
    missing = df.isnull().sum()  # Count nulls per column
    print("\nMissing values per column:")
    if missing.sum() == 0:
        # No missing values found
        print("  None")
    else:
        # Print only columns that have at least one missing value
        print(missing[missing > 0].to_string())

    # Show the first 3 rows for a quick data sanity check
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print()  # Blank line between datasets
