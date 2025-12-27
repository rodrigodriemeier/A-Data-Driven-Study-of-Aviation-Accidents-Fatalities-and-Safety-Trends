import pandas as pd

# Load raw data
db = pd.read_csv("aviation-accident-data-2023-05-16.csv")

# -------------------------------
# Filtering and cleaning
# -------------------------------

# Drop exact duplicates
db = db.drop_duplicates()

# Drop columns not used in the analysis
cols_to_drop = ["location", "registration", "operator"]
db.drop(columns=cols_to_drop, inplace=True)

# Drop date column if it adds no extra yearly information
cond = (db["year"].isna()) & (db["date"] != "date unk.")
if cond.sum() == 0:
    db.drop("date", axis=1, inplace=True)

# Standardize unknown year
db.loc[db["year"] == "unknown", "year"] = pd.NA

# Standardize unknown country
db.loc[(db["country"] == "?") | (db["country"] == "Unknown country"), "country"] = pd.NA

# Convert year to integer
db["year"] = db["year"].astype("Int64")

# Handle fatalities (sum values like "1+2")
db["fatalities"] = db["fatalities"].str.split("+")

for idx, fatality in db["fatalities"].items():
    if isinstance(fatality, list):
        if pd.notna(fatality).all():
            db.at[idx, "fatalities"] = sum(int(i) for i in fatality)

db["fatalities"] = db["fatalities"].astype("Int64")

# Save cleaned dataset
db.to_csv("aviation_cleaned.csv", index=False)

print("Cleaning completed. Cleaned dataset saved.")
