import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# LOAD CLEANED DATA
# ===============================
db = pd.read_csv("aviation_cleaned.csv")

# ===============================
# BASIC TEMPORAL ANALYSIS
# ===============================
accidents_per_year = db.groupby("year").size().reset_index(name="accidents_per_year")
fatalities_per_year = db.groupby("year")["fatalities"].sum().reset_index(name="fatalities_per_year")

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(accidents_per_year["year"], accidents_per_year["accidents_per_year"], color="darkcyan")
axs[0].set_title("Accidents Per Year")
axs[0].set_ylabel("Accidents")
axs[0].grid(alpha=0.3)

axs[1].plot(fatalities_per_year["year"], fatalities_per_year["fatalities_per_year"], color="darkcyan")
axs[1].set_title("Fatalities Per Year")
axs[1].set_xlabel("Year")
axs[1].set_ylabel("Fatalities")
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# WORLD WAR II ANALYSIS
# ===============================
db_ww2 = db[(db["year"] >= 1939) & (db["year"] <= 1945)]
top5_airplanes = db_ww2["type"].value_counts(normalize=True).head(5) * 100

plt.figure(figsize=(12, 8))
plt.bar(top5_airplanes.index, top5_airplanes.values, color="darkcyan")
plt.title("Top 5 Aircraft Models in Accidents (1939–1945)")
plt.ylabel("Frequency (%)")
plt.grid(alpha=0.3)
plt.show()

db_1944 = db[db["year"] == 1944]
top5_countries = db_1944["country"].value_counts(normalize=True).head(5) * 100

plt.figure(figsize=(10, 8))
plt.bar(top5_countries.index, top5_countries.values, color="darkcyan")
plt.title("Top 5 Accident Locations – 1944")
plt.ylabel("Accidents (%)")
plt.grid(alpha=0.3)
plt.show()

# ===============================
# UNLAWFUL INTERFERENCE (H)
# ===============================
db_hijack = db.replace({"H1": "H", "H2": "H"})
cond_modern = db_hijack["year"] >= 1980

h_cat = (
    db_hijack[cond_modern & (db_hijack["cat"] == "H")]
    .groupby("year")
    .size()
)
h_cat_prop = (h_cat / db_hijack[cond_modern].groupby("year").size() * 100).fillna(0)

h_fat = (
    db_hijack[cond_modern & (db_hijack["cat"] == "H")]
    .groupby("year")["fatalities"]
    .sum()
)
h_fat_prop = (h_fat / db_hijack[cond_modern].groupby("year")["fatalities"].sum()).fillna(0) * 100

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].bar(h_cat_prop.index, h_cat_prop.values, color="darkcyan")
axs[0].set_title("H Accidents Share (%)")
axs[0].grid(alpha=0.3)

axs[1].bar(h_fat_prop.index, h_fat_prop.values, color="darkcyan")
axs[1].set_title("H Fatalities Share (%)")
axs[1].set_xlabel("Year")
axs[1].grid(alpha=0.3)

plt.show()

# ===============================
# SEVERITY SCORE
# ===============================
severity = fatalities_per_year.copy()
severity["severity_score"] = (
    fatalities_per_year["fatalities_per_year"] /
    accidents_per_year["accidents_per_year"]
)
severity["severity_score"] = severity["severity_score"] / severity["severity_score"].max() * 100

sev_modern = severity[severity["year"] >= 1990]
coef = np.polyfit(sev_modern["year"], sev_modern["severity_score"], 1)
trend = np.poly1d(coef)

plt.figure(figsize=(10, 8))
plt.plot(sev_modern["year"], sev_modern["severity_score"], color="darkcyan")
plt.plot(sev_modern["year"], trend(sev_modern["year"]), "--", color="red")
plt.title("Severity Score Trend (1990–2023)")
plt.xlabel("Year")
plt.ylabel("Severity Score")
plt.grid(alpha=0.3)
plt.show()

# ===============================
# ACCIDENT TYPE BY DECADE
# ===============================
db_att_cat = db.copy()
db_att_cat["cat"] = db_att_cat["cat"].replace({
    "A1": "A", "A2": "A",
    "O1": "O", "O2": "O",
    "H1": "H", "H2": "H",
    "C1": "C", "C2": "C",
    "I1": "I", "I2": "I",
    "U1": "U"
})

db_att_cat["decade"] = (db_att_cat["year"] // 10) * 10

df = (
    db_att_cat.groupby("decade")["cat"]
    .value_counts(normalize=True)
    .mul(100)
    .unstack()
    .fillna(0)
)

plt.figure(figsize=(12, 8))
bottom = np.zeros(len(df))

for c in df.columns:
    plt.bar(df.index, df[c], bottom=bottom, label=c)
    bottom += df[c].values

plt.title("Accident Type Distribution by Decade")
plt.xlabel("Decade")
plt.ylabel("Proportion (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
