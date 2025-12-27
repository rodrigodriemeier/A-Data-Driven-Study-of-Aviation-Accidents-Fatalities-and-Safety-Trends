import pandas as pd

# Load cleaned data
db = pd.read_csv("aviation_cleaned.csv")

# -------------------------------
# Reliability checks
# -------------------------------

print(db.head())

# Category vs fatalities (high and low extremes)
test_cat_fat_top = db.sort_values(by="fatalities", ascending=False).head(5)
test_cat_fat_low = db[db["fatalities"] == 0]

print(test_cat_fat_top.loc[:, ["fatalities", "cat"]])
print(test_cat_fat_low.loc[:, ["fatalities", "cat"]])

# Check whether H1/H2 are coherent as event indicators
test_cat_valid_for_event = db[(db["cat"] == "H1") | (db["cat"] == "H2")]
print(test_cat_valid_for_event.loc[:, ["year", "type"]])

# Years with no accidents
accidents_per_year = db.groupby("year").size().reset_index(name="accidents_per_year")

count_zero = 0
for v in accidents_per_year["accidents_per_year"]:
    if v == 0:
        count_zero += 1

print(f"We found {count_zero} years with no accidents")

# NA fatalities per year
nas_fat_per_year = (
    db["fatalities"]
    .isna()
    .groupby(db["year"])
    .sum()
    .reset_index(name="quantity_of_NAs")
    .sort_values(by="quantity_of_NAs")
    .tail(10)
)

print(nas_fat_per_year)

# Check for absurd fatality values
print(
    db.dropna(subset=["fatalities"])
    .sort_values(by="fatalities")
    .tail(10)
    .loc[:, ["year", "type", "fatalities"]]
)

print("Reliability checks completed.")
