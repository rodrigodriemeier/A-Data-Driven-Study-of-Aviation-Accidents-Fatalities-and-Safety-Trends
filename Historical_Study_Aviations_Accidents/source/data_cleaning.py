import pandas as pd

# ===============================
# LOAD RAW DATA (ONLY HERE)
# ===============================
db = pd.read_csv("aviation-accident-data-2023-05-16.csv")

# ===============================
# DATA CLEANING
# ===============================

# Drop exact duplicates
db = db.drop_duplicates()

# Drop unused columns
db.drop(columns=["location", "registration", "operator"], inplace=True)

# Drop date if it adds no information beyond year
cond = (db["year"].isna()) & (db["date"] != "date unk.")
if cond.sum() == 0:
    db.drop("date", axis=1, inplace=True)

# Standardize unknown values
db.loc[db["year"] == "unknown", "year"] = pd.NA
db.loc[(db["country"] == "?") | (db["country"] == "Unknown country"), "country"] = pd.NA

# Convert year to integer
db["year"] = db["year"].astype("Int64")

# Fatalities: handle cases like "1+2"
db["fatalities"] = db["fatalities"].str.split("+")

for idx, fatality in db["fatalities"].items():
    if isinstance(fatality, list) and pd.notna(fatality).all():
        db.at[idx, "fatalities"] = sum(int(i) for i in fatality)

db["fatalities"] = db["fatalities"].astype("Int64")

# ===============================
# SAVE CLEAN DATASET
# ===============================
db.to_csv("aviation_cleaned.csv", index=False)

print("Cleaned dataset saved as aviation_cleaned.csv")
print(db.head())
