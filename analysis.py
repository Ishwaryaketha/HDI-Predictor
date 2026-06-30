import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("hdi.csv")

# Convert GNI column to numeric
df["Gross National Income (GNI) per Capita"] = (
    df["Gross National Income (GNI) per Capita"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Gross National Income (GNI) per Capita"] = pd.to_numeric(
    df["Gross National Income (GNI) per Capita"],
    errors="coerce"
)

df = df.apply(pd.to_numeric, errors="ignore")
df = df.dropna()

# Create folder if it doesn't exist
os.makedirs("static/images", exist_ok=True)

# 1. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/images/heatmap.png")
plt.close()

# 2. HDI Distribution
plt.figure(figsize=(7,5))
sns.histplot(df["Human Development Index (HDI)"], bins=20, kde=True)
plt.title("HDI Distribution")
plt.tight_layout()
plt.savefig("static/images/distribution.png")
plt.close()

# 3. Scatter Plot
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Gross National Income (GNI) per Capita",
    y="Human Development Index (HDI)"
)
plt.title("HDI vs GNI")
plt.tight_layout()
plt.savefig("static/images/scatter.png")
plt.close()

# 4. Pair Plot
pair = sns.pairplot(
    df[[
        "Life Expectancy at Birth",
        "Expected Years of Education",
        "Mean Years of Education",
        "Gross National Income (GNI) per Capita",
        "Human Development Index (HDI)"
    ]],
    height=3
)

pair.fig.suptitle("HDI Pair Plot", y=1.02)

pair.savefig("static/images/pairplot.png")

plt.close('all')

# 5. Strip Plot
plt.figure(figsize=(10,5))

sns.stripplot(
    x=df["Human Development Index (HDI)"],
    color="royalblue"
)

plt.title("HDI Strip Plot", fontsize=16)
plt.xlabel("Human Development Index (HDI)")

plt.tight_layout()

plt.savefig("static/images/stripplot.png")

plt.close()

print("Analysis graphs created successfully!")