# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# Load Adult dataset
columns = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

df = pd.read_csv('../data/adult.data', header=None, names=columns, na_values=' ?', skipinitialspace=True)


print(df.head())
print(df.info())
print(df.describe())

# =========================================================
# Task 1: Data Quality Check
# =========================================================

# Missing values
missing = df.isnull().sum()
print("Missing values:\n", missing)

# Duplicates
duplicates = df.duplicated().sum()
print("Duplicates:", duplicates)

# Visualize outliers in 'hours-per-week'
sns.boxplot(x=df['hours-per-week'])
plt.show()

# =========================================================
# Task 2: Data Cleaning
# =========================================================

# Fill missing categorical with mode
imputer_cat = SimpleImputer(strategy='most_frequent')
df[df.select_dtypes(include=['object']).columns] = imputer_cat.fit_transform(
    df.select_dtypes(include=['object'])
)

# Drop rows with too many missing (if any)
df.dropna(inplace=True)

# Detect outliers in 'hours-per-week'
Q1 = df['hours-per-week'].quantile(0.25)
Q3 = df['hours-per-week'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['hours-per-week'] < (Q1 - 1.5 * IQR)) | (df['hours-per-week'] > (Q3 + 1.5 * IQR))]
print("Outliers:\n", outliers)

# Remove outliers
df = df[~((df['hours-per-week'] < (Q1 - 1.5 * IQR)) | (df['hours-per-week'] > (Q3 + 1.5 * IQR)))]

# Discretize 'age' into bins
df['age_binned'] = pd.cut(df['age'], bins=[0,25,45,65,120],
                          labels=['Young','Adult','Middle-aged','Senior'])

# =========================================================
# Task 3: Data Transformation and Discretization
# =========================================================

# Normalization - MinMax for 'age'
scaler_minmax = MinMaxScaler()
df['age_minmax'] = scaler_minmax.fit_transform(df[['age']])

# Z-score for 'hours-per-week'
scaler_z = StandardScaler()
df['hours_zscore'] = scaler_z.fit_transform(df[['hours-per-week']])

# Decimal scaling for 'fnlwgt'
max_val = df['fnlwgt'].max()
j = len(str(int(max_val)))
df['fnlwgt_decimal'] = df['fnlwgt'] / (10 ** j)

print("After Normalization:\n", df.head())

# Equal-width binning (education-num)
df['education_width'] = pd.cut(df['education-num'], bins=3,
                               labels=['Low','Medium','High'])

# Equal-depth binning (capital-gain)
df['capital_gain_depth'] = pd.qcut(df['capital-gain'], q=3,
                                   labels=['Low','Medium','High'])

# Concept hierarchy for age
bins = [0,18,35,60,np.inf]
labels = ['Youth','Young Adult','Adult','Senior']
df['age_hierarchy'] = pd.cut(df['age'], bins=bins, labels=labels)

print("After Discretization:\n", df.head())

# =========================================================
# Task 4: Data Reduction
# =========================================================

# PCA on numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df[numerical_cols]), columns=['PC1','PC2'])
print("Explained Variance:", pca.explained_variance_ratio_)

# Feature selection: drop irrelevant (example: 'fnlwgt')
df.drop(['fnlwgt'], axis=1, inplace=True)

print("After Feature Selection:\n", df.head())

# Sampling (50%)
sample_df = df.sample(frac=0.5, random_state=42)

# Histogram (hours-per-week)
df['hours-per-week'].hist(bins=10)
plt.show()

# Clustering (Age groups)
kmeans = KMeans(n_clusters=3, n_init=10)
df['age_cluster'] = kmeans.fit_predict(df[['age']])
print(df[['age','age_cluster']].head())
