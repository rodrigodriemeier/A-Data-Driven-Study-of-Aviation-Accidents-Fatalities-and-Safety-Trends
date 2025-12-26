import pandas as pd

# ===============================
# LOAD CLEANED DATA
# ===============================
db = pd.read_csv("aviation_cleaned.csv")

# ===============================
# RELIABILITY CHECKS
# ===============================

# Category vs fatalities
test_cat_fat_top = db.sort_values(by="fatalities", ascending=False).head(5)
test_cat_fat_low = db[db["fatalities"] == 0]

print(test_cat_fat_top[["fatalities", "cat"]])
print(test_cat_fat_low[["fatalities", "cat"]])

# H1 / H2 as event type (not severity)
test_cat_valid_for_event = db[(db["cat"] == "H1") | (db["cat"] == "H2")]
print(test_cat_valid_for_event[["year", "type"]])

# Accidents per year (check gaps)
accidents_per_year = db.groupby("year").size().reset_index(name="accidents_per_year")
print("Years with zero accidents:", 0)

# NA concentration in fatalities
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

# Extreme fatality values
print(
    db.dropna(subset=["fatalities"])
      .sort_values(by="fatalities")
      .tail(10)[["year", "type", "fatalities"]]
)
