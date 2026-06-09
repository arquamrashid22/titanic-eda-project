# =========================
# TITANIC EDA PROJECT
# =========================

# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1️⃣ LOAD DATA
# =========================

df = sns.load_dataset("titanic")

# =========================
# 2️⃣ DATA OVERVIEW
# =========================

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# =========================
# 3️⃣ MISSING VALUES CHECK
# =========================

print(df.isnull().sum())

# =========================
# 4️⃣ DATA CLEANING
# =========================

# Fill missing age with median
df["age"] = df["age"].fillna(df["age"].median())

# Fill embark_town with mode
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# =========================
# 5️⃣ EXPLORATORY DATA ANALYSIS
# =========================

sns.set_style("whitegrid")

# Survival count
plt.figure()
sns.countplot(x="survived", data=df)
plt.title("Survival Count")
plt.savefig("images/survival_count.png")
plt.show()

# Gender vs Survival
plt.figure()
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Gender vs Survival")
plt.savefig("images/gender_survival.png")
plt.show()

# Class vs Survival
plt.figure()
sns.countplot(x="class", hue="survived", data=df)
plt.title("Class vs Survival")
plt.savefig("images/class_survival.png")
plt.show()

# Age distribution
plt.figure()
sns.histplot(df["age"], bins=20)
plt.title("Age Distribution")
plt.savefig("images/age_distribution.png")
plt.show()

# Fare distribution
plt.figure()
sns.histplot(df["fare"], bins=20)
plt.title("Fare Distribution")
plt.savefig("images/fare_distribution.png")
plt.show()

# Scatter plot (Age vs Fare)
plt.figure()
sns.scatterplot(x="age", y="fare", hue="survived", data=df)
plt.title("Age vs Fare")
plt.savefig("images/age_fare_scatter.png")
plt.show()

# =========================
# 6️⃣ CORRELATION ANALYSIS
# =========================

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# =========================
# 7️⃣ GROUP ANALYSIS
# =========================

print("\nSurvival Rate by Gender:")
print(df.groupby("sex")["survived"].mean())

print("\nSurvival Rate by Class:")
print(df.groupby("class")["survived"].mean())

print("\nAverage Fare by Class & Sex:")
print(df.groupby(["class","sex"])["fare"].mean())

# =========================
# 8️⃣ PIVOT TABLE
# =========================

print("\nPivot Table (Fare):")
print(
    df.pivot_table(
        values="fare",
        index="class",
        columns="sex",
        aggfunc="mean"
    )
)

# =========================
# 9️⃣ FINAL DASHBOARD
# =========================

fig, ax = plt.subplots(2,2, figsize=(12,8))

sns.histplot(df["age"], ax=ax[0,0])
ax[0,0].set_title("Age Distribution")

sns.boxplot(x="class", y="age", data=df, ax=ax[0,1])
ax[0,1].set_title("Class vs Age")

sns.scatterplot(x="age", y="fare", hue="survived", data=df, ax=ax[1,0])
ax[1,0].set_title("Age vs Fare")

sns.countplot(x="class", data=df, ax=ax[1,1])
ax[1,1].set_title("Passenger Class Count")

plt.tight_layout()
plt.savefig("images/dashboard.png")
plt.show()

# =========================
# END OF PROJECT
# =========================