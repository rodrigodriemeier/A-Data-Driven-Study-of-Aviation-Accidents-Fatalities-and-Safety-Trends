import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD CLEAN DATASET
# =========================
db = pd.read_csv("aviation_cleaned.csv")

# Ensure numeric types (no cleaning logic, just type safety for plots/filters)
db["year"] = pd.to_numeric(db["year"], errors="coerce")
db["fatalities"] = pd.to_numeric(db["fatalities"], errors="coerce")

# =========================
# 1) ACCIDENTS / FATALITIES PER YEAR + TOP 5
# =========================
accidents_per_year = db.groupby("year").size().reset_index(name="accidents_per_year")
fatalities_per_year = db.groupby("year")["fatalities"].sum().reset_index(name="fatalities_per_year")

peak_accidents_years = accidents_per_year.sort_values(by="accidents_per_year", ascending=False).head(5)
peak_fatalities_years = fatalities_per_year.sort_values(by="fatalities_per_year", ascending=False).head(5)

plt.figure(figsize=(10, 8))
plt.plot(accidents_per_year["year"], accidents_per_year["accidents_per_year"], color="darkcyan")
plt.title("Accidents Per Year")
plt.xlabel("Year")
plt.ylabel("Accidents per year")
plt.grid(alpha=0.3)

plt.figure(figsize=(10, 8))
plt.plot(fatalities_per_year["year"], fatalities_per_year["fatalities_per_year"], color="darkcyan")
plt.title("Fatalities Per Year")
plt.xlabel("Year")
plt.ylabel("Fatalities per year")
plt.grid(alpha=0.3)

plt.figure(figsize=(10, 8))
plt.bar(peak_accidents_years["year"].astype("Int64").astype(str), peak_accidents_years["accidents_per_year"], color="darkcyan")
plt.title("Top 5 Years With Most Accidents")
plt.xlabel("Year")
plt.ylabel("Accidents")
plt.grid(alpha=0.3)

plt.figure(figsize=(10, 8))
plt.bar(peak_fatalities_years["year"].astype("Int64").astype(str), peak_fatalities_years["fatalities_per_year"], color="darkcyan")
plt.title("Top 5 Years With Most Fatalities")
plt.xlabel("Year")
plt.ylabel("Fatalities")
plt.grid(alpha=0.3)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(accidents_per_year["year"], accidents_per_year["accidents_per_year"], color="darkcyan")
axs[0].set_title("Accidents Per Year")
axs[0].set_ylabel("Number of accidents")
axs[0].grid(alpha=0.3)

axs[1].plot(fatalities_per_year["year"], fatalities_per_year["fatalities_per_year"], color="darkcyan")
axs[1].set_title("Fatalities Per Year")
axs[1].set_xlabel("Year")
axs[1].set_ylabel("Number of fatalities")
axs[1].grid(alpha=0.3)

plt.tight_layout()

# =========================
# 2) WWII ANALYSIS (1939–1945) + PEAK YEAR 1944
# =========================
db_ww2 = db[(db["year"] >= 1939) & (db["year"] <= 1945)]

top5_airplanes = db_ww2["type"].value_counts(normalize=True).head(5) * 100

plt.figure(figsize=(13, 8))
plt.bar(top5_airplanes.index, top5_airplanes.values, color="darkcyan")
plt.title("Top 5 Models Of Aircraft Involved in Accidents (%)\n(1939 - 1945)")
plt.xlabel("Aircraft model")
plt.ylabel("Frequency (%)")
plt.grid(alpha=0.3)

db_1944 = db[db["year"] == 1944]
top5_countries_acc = db_1944["country"].value_counts(normalize=True).head(5) * 100

plt.figure(figsize=(10, 8))
plt.bar(top5_countries_acc.index, top5_countries_acc.values, color="darkcyan")
plt.title("Top 5 Locations Of Where Did Most Of The Accidents Happen During The Peak Year (1944) (%)")
plt.xlabel("Country")
plt.ylabel("Accidents frequency (%)")
plt.grid(alpha=0.3)

# =========================
# 3) H CATEGORY (H1+H2 -> H) (1980+): % of accidents + % of fatalities (subplots)
# =========================
db_hijack = db.copy()
db_hijack["cat"] = db_hijack["cat"].replace({"H1": "H", "H2": "H"})

cond_modern_times = db_hijack["year"] >= 1980

# % of H accidents among all accidents per year
h_acc_quant = db_hijack[cond_modern_times & (db_hijack["cat"] == "H")].groupby("year").size()
all_acc_quant = db_hijack[cond_modern_times].groupby("year").size()

db_hijack_analysis_cat_prop = (h_acc_quant / all_acc_quant * 100).reset_index(name="prop_cat_H")

# % of H fatalities among total fatalities per year
h_fat_quant = db_hijack[cond_modern_times & (db_hijack["cat"] == "H")].groupby("year")["fatalities"].sum()
all_fat_quant = db_hijack[cond_modern_times].groupby("year")["fatalities"].sum()

db_hijack_analysis_fat_prop = (h_fat_quant / all_fat_quant).reset_index(name="prop_fat_H")
db_hijack_analysis_fat_prop["prop_fat_H"] = db_hijack_analysis_fat_prop["prop_fat_H"].fillna(0) * 100

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].bar(db_hijack_analysis_cat_prop["year"], db_hijack_analysis_cat_prop["prop_cat_H"], color="darkcyan")
axs[0].set_title("Percentage of H Accidents Among All Accidents per Year\n(1980 - 2021)")
axs[0].set_ylabel("H Accidents Participation (%)")
axs[0].set_ylim(0, 75)
axs[0].grid(alpha=0.3)

axs[1].bar(db_hijack_analysis_fat_prop["year"], db_hijack_analysis_fat_prop["prop_fat_H"], color="darkcyan")
axs[1].set_title("Percentage of H Accidents Fatalities Among Total Fatalities per Year\n(1980 - 2021)")
axs[1].set_ylabel("H Accidents Fatalities Participation(%)")
axs[1].set_xlabel("Years")
axs[1].grid(alpha=0.3)

# Keep your annotation (position may be adjusted manually if needed)
axs[1].annotate(
    "2001 – September 11 attacks",
    xy=(2001, float(db_hijack_analysis_fat_prop.loc[db_hijack_analysis_fat_prop["year"] == 2001, "prop_fat_H"].fillna(0).max())),
    xytext=(1987, 60),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    color="red"
)

plt.tight_layout()

# =========================
# 4) TRENDS (1990+) FOR ACCIDENTS & FATALITIES (subplots)
# =========================
cond_after_1990_acc = accidents_per_year["year"] >= 1990
cond_after_1990_fat = fatalities_per_year["year"] >= 1990

acc_modern = accidents_per_year[cond_after_1990_acc].dropna(subset=["year"])
fat_modern = fatalities_per_year[cond_after_1990_fat].dropna(subset=["year"])

coef_acc = np.polyfit(acc_modern["year"], acc_modern["accidents_per_year"], 1)
coef_fat = np.polyfit(fat_modern["year"], fat_modern["fatalities_per_year"], 1)

trend_acc = np.poly1d(coef_acc)
trend_fat = np.poly1d(coef_fat)

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(acc_modern["year"], acc_modern["accidents_per_year"], color="darkcyan")
axs[0].plot(acc_modern["year"], trend_acc(acc_modern["year"]), linestyle="--", color="red")
axs[0].annotate(
    f"    Tendency Line\ny = {coef_acc[0]:.1f}x + {coef_acc[1]:.1f}",
    xy=(2001, 234),
    xytext=(2001, 150),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    color="red"
)
axs[0].set_title("Accidents Tendency (1990 - 2021)")
axs[0].set_ylabel("Accidents")
axs[0].grid(alpha=0.3)

axs[1].plot(fat_modern["year"], fat_modern["fatalities_per_year"], color="darkcyan")
axs[1].plot(fat_modern["year"], trend_fat(fat_modern["year"]), linestyle="--", color="red")
axs[1].annotate(
    f"       Tendency Line\ny = {coef_fat[0]:.1f}x + {coef_fat[1]:.1f}",
    xy=(2001, 1496),
    xytext=(2006, 1490),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    color="red"
)
axs[1].set_title("Fatalities Tendency (1990 - 2021)")
axs[1].set_xlabel("Years")
axs[1].set_ylabel("Fatalities")
axs[1].grid(alpha=0.3)

plt.tight_layout()

# =========================
# 5) SEVERITY SCORE (1919–2023) + (1990–2023) with trend
# =========================
df_year = pd.merge(accidents_per_year, fatalities_per_year, on="year", how="inner")

severity = df_year[["year"]].copy()
severity["severity_score"] = df_year["fatalities_per_year"] / df_year["accidents_per_year"]
severity["severity_score"] = (severity["severity_score"] / severity["severity_score"].max()) * 100

severity_modern = severity[severity["year"] >= 1990].dropna(subset=["year", "severity_score"])

coef_sev = np.polyfit(severity_modern["year"], severity_modern["severity_score"], 1)
trend_sev = np.poly1d(coef_sev)

plt.figure(figsize=(10, 8))
plt.plot(severity["year"], severity["severity_score"], color="darkcyan")
plt.title("Severity Score Over the Years (1919 - 2023)")
plt.xlabel("Year")
plt.ylabel("Severity Score")
plt.grid(alpha=0.3)

plt.figure(figsize=(10, 8))
plt.plot(severity_modern["year"], severity_modern["severity_score"], color="darkcyan")
plt.plot(severity_modern["year"], trend_sev(severity_modern["year"]), linestyle="--", color="red")
plt.title("Severity Score Over the Years (1990 - 2023)")
plt.xlabel("Year")
plt.ylabel("Severity Score")
plt.grid(alpha=0.3)

plt.annotate(
    f"    Tendency Line\ny = {coef_sev[0]:.1f}x + {coef_sev[1]:.1f}",
    xy=(2001, 33.5),
    xytext=(2009, 60),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
    color="red"
)

# =========================
# 6) ACCIDENT TYPE DISTRIBUTION BY DECADE (stacked) + ZOOMED
# =========================
db_att_cat = db.copy()
db_att_cat["cat"] = db_att_cat["cat"].replace({
    "U1": "U",
    "A1": "A", "A2": "A",
    "O1": "O", "O2": "O",
    "H1": "H", "H2": "H",
    "C1": "C", "C2": "C",
    "I1": "I", "I2": "I"
})

db_att_cat["decade"] = (db_att_cat["year"] // 10) * 10

df_dec = (
    db_att_cat.groupby("decade")["cat"]
    .value_counts(normalize=True)
    .mul(100)
    .reset_index(name="cat_prop_dec")
)

df_plot = (
    df_dec.pivot(index="decade", columns="cat", values="cat_prop_dec")
    .fillna(0)
)

# Ensure all expected columns exist
for c in ["A", "O", "H", "C", "I", "U"]:
    if c not in df_plot.columns:
        df_plot[c] = 0.0

df_plot = df_plot[["A", "O", "H", "C", "I", "U"]]

plt.figure(figsize=(12, 8))
plt.bar(df_plot.index, df_plot["A"], label="Accidents (A)")
plt.bar(df_plot.index, df_plot["O"], bottom=df_plot["A"], label="Operational (O)")
plt.bar(df_plot.index, df_plot["H"], bottom=df_plot["A"] + df_plot["O"], label="Unlawful Interference (H)")
plt.bar(df_plot.index, df_plot["C"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"], label="Criminal Acts (C)")
plt.bar(df_plot.index, df_plot["I"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"] + df_plot["C"], label="Incidents (I)")
plt.bar(df_plot.index, df_plot["U"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"] + df_plot["C"] + df_plot["I"], label="Unknown (U)")
plt.xlabel("Decade")
plt.ylabel("Proportion of Accidents (%)")
plt.title("Distribution of Accident Types by Decade")
plt.legend()
plt.grid(alpha=0.3)

# Zoomed version (same stacked chart, just zoom in Y)
plt.figure(figsize=(12, 8))
plt.bar(df_plot.index, df_plot["A"], label="Accidents (A)")
plt.bar(df_plot.index, df_plot["O"], bottom=df_plot["A"], label="Operational (O)")
plt.bar(df_plot.index, df_plot["H"], bottom=df_plot["A"] + df_plot["O"], label="Unlawful Interference (H)")
plt.bar(df_plot.index, df_plot["C"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"], label="Criminal Acts (C)")
plt.bar(df_plot.index, df_plot["I"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"] + df_plot["C"], label="Incidents (I)")
plt.bar(df_plot.index, df_plot["U"], bottom=df_plot["A"] + df_plot["O"] + df_plot["H"] + df_plot["C"] + df_plot["I"], label="Unknown (U)")
plt.xlabel("Decade")
plt.ylabel("Proportion of Accidents (%)")
plt.title("Distribution of Accident Types by Decade (Zoomed View)")
plt.ylim(75, 100)
plt.legend()
plt.grid(alpha=0.3)

# =========================
# SHOW ALL FIGURES
# =========================
plt.show()
