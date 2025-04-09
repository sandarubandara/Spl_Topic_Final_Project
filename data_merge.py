import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('crop_yield.csv')
df2 = pd.read_csv('machinery_use.csv')
df3 = pd.read_csv('labour_and_land_index.csv')

# Standardize column names
df1.columns = df1.columns.str.strip().str.title()
df2.columns = df2.columns.str.strip().str.title()
df3.columns = df3.columns.str.strip().str.title()

# Merge based on 'Year' and 'Country' columns
temp_merge = pd.merge(df1, df2, on=["Year", "Country"], how="inner")
merged_df = pd.merge(temp_merge, df3, on=["Year", "Country"], how="inner")

# Save the final merged DataFrame to CSV
merged_df.to_csv('final_crop_analysis.csv', index=False)

# Display the first few rows
print(merged_df.head())