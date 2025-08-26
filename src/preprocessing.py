# 0. Imports & settings
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# (Optional) display options
pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 140)

# 1. Column names (from adult.names)
columns = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

# 2. Read the dataset from your local folder
file_path = "data/adult.data"   # make sure this path is correct
df = pd.read_csv(file_path, header=None, names=columns,
                 na_values=' ?', skipinitialspace=True)

print("Initial shape:", df.shape)
print(df.head(6))

# 3. Quick exploratory summary
print("\nColumn types and non-null counts:")
print(df.info())
print("\nBasic stats (numerical):")
print(df.describe())

# 4. Missing values check
print("\nMissing values per column:")
print(df.isna().sum())

# 5. Clean whitespace & uniform text (good hygiene for categorical data)
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].str.strip()

# 6. Target encoding (income) -> 0 / 1
df['income'] = df['income'].apply(lambda x: 1 if isinstance(x, str) and '>50' in x else 0)

# 7. Split feature lists
numeric_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
categorical_cols = [c for c in df.columns if c not in numeric_cols + ["income"]]

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# 8. Handle missing values
# Approach A: drop rows with missing
df_dropna = df.dropna()
print("\nAfter dropna shape:", df_dropna.shape)

# Approach B: impute categorical missing with mode
df_impute = df.copy()
cat_imputer = SimpleImputer(strategy="most_frequent")
df_impute[categorical_cols] = cat_imputer.fit_transform(df_impute[categorical_cols])
print("After mode-imputation, missing counts:")
print(df_impute.isna().sum())

# We'll continue with df_impute
df = df_impute

# 9. Log-transform skewed features
for col in ["capital-gain", "capital-loss"]:
    df[col + "_log1p"] = np.log1p(df[col])   # log(1+x)

# 10. Age binning
age_bins = [0, 25, 45, 65, 120]
age_labels = ["young", "adult", "middle-aged", "senior"]
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

# 11. One-hot encode categoricals
df_encoded = pd.get_dummies(df.drop(columns=["income"]), 
                            columns=categorical_cols + ["age_group"],
                            drop_first=True)

print("\nAfter one-hot encoding shape:", df_encoded.shape)

# 12. Scale numeric features
num_for_scaling = ["age","fnlwgt","education-num","hours-per-week","capital-gain_log1p","capital-loss_log1p"]
scaler = StandardScaler()
df_encoded[num_for_scaling] = scaler.fit_transform(df_encoded[num_for_scaling])

# 13. PCA to 95% variance
pca = PCA(n_components=0.95, svd_solver='full')
X_reduced = pca.fit_transform(df_encoded.values)
print("\nOriginal feature count:", df_encoded.shape[1])
print("Reduced feature count via PCA (95% var):", X_reduced.shape[1])

# 14. Final dataset
X_final = pd.DataFrame(X_reduced, index=df_encoded.index,
                       columns=[f"PC{i+1}" for i in range(X_reduced.shape[1])])
y_final = df['income'].loc[X_final.index].reset_index(drop=True)
X_final = X_final.reset_index(drop=True)

print("\nFinal dataset shapes -> X:", X_final.shape, "y:", y_final.shape)

# 15. Save to CSV
out_path = "adult_preprocessed_pca.csv"
pd.concat([X_final, y_final.rename("income")], axis=1).to_csv(out_path, index=False)
print(f"\nSaved preprocessed dataset to {os.path.abspath(out_path)}")
