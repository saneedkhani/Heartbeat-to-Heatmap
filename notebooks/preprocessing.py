"""
DS-3002 Data Mining — Assignment #4
Preprocessing & Data Setup (12 Marks)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATH SETUP — always resolves relative to THIS file
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../notebooks/
ROOT_DIR   = os.path.dirname(BASE_DIR)                    # .../Assignment 04/

DATA_PATH   = os.path.join(ROOT_DIR, 'data',    'processed_cleveland.data')
MODELS_DIR  = os.path.join(ROOT_DIR, 'models')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PRE-1  Load the dataset
# ─────────────────────────────────────────────
COLUMN_NAMES = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, na_values='?')

print("=" * 60)
print("PRE-1: Dataset Shape and Preview")
print("=" * 60)
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# ─────────────────────────────────────────────
# PRE-2  Missing values
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRE-2: Missing Values")
print("=" * 60)
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print(f"\nRows with missing values: {df.isnull().any(axis=1).sum()}")
df.dropna(inplace=True)
print(f"Retained rows after dropping NaN: {len(df)}")

# Binarise target: 0 = no disease, 1 = disease
df['target'] = (df['target'] > 0).astype(int)

# ─────────────────────────────────────────────
# PRE-3  Class distribution
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRE-3: Class Distribution")
print("=" * 60)
counts = df['target'].value_counts()
pct    = df['target'].value_counts(normalize=True) * 100
dist   = pd.DataFrame({'Count': counts, 'Percent (%)': pct.round(2)})
dist.index = ['No Disease (0)', 'Disease (1)']
print(dist)
ratio    = counts.min() / counts.max()
balanced = ratio >= 0.75
print(f"\nMinority/Majority ratio: {ratio:.2f} — {'Balanced' if balanced else 'Imbalanced'}")
print("Decision: Apply SMOTE on training split only.")

# ─────────────────────────────────────────────
# PRE-4  Encoding & Scaling definitions
# ─────────────────────────────────────────────
CATEGORICAL = ['cp', 'restecg', 'slope', 'thal']
CONTINUOUS  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# ─────────────────────────────────────────────
# PRE-5  Stratified 80/20 split (random_state=42)
# ─────────────────────────────────────────────
X_raw = df.drop('target', axis=1)
y     = df['target']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=42
)

# One-hot encode categoricals
X_train_enc = pd.get_dummies(X_train_raw, columns=CATEGORICAL)
X_test_enc  = pd.get_dummies(X_test_raw,  columns=CATEGORICAL)
X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# Standardise continuous columns (fit on train only)
scaler = StandardScaler()
X_train_enc[CONTINUOUS] = scaler.fit_transform(X_train_enc[CONTINUOUS])
X_test_enc[CONTINUOUS]  = scaler.transform(X_test_enc[CONTINUOUS])

# SMOTE on training split only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

print("\n" + "=" * 60)
print("PRE-5: Train/Test Split Sizes")
print("=" * 60)
print(f"X_train (before SMOTE): {X_train_enc.shape}")
print(f"X_train (after  SMOTE): {X_train_sm.shape}")
print(f"X_test               : {X_test_enc.shape}")
print(f"y_train class dist after SMOTE: {pd.Series(y_train_sm).value_counts().to_dict()}")

# ─────────────────────────────────────────────
# PRE-6  Correlation heatmap (original numeric)
# ─────────────────────────────────────────────
numeric_cols = ['age','trestbps','chol','thalach','exang','oldpeak','ca','target']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('PRE-6: Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'pre_correlation_heatmap.png'), dpi=150)
plt.close()

# Top 3 correlated pairs
pairs = corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
pairs = pairs.stack().abs().sort_values(ascending=False)
print("\n" + "=" * 60)
print("PRE-6: Top 3 Feature Pairs by Absolute Correlation")
print("=" * 60)
for (c1, c2), v in pairs.head(3).items():
    print(f"  {c1} <-> {c2}: {v:.3f}")
print("""
Naive Bayes note: NB assumes feature independence.
High correlations (e.g., thalach<->age, exang<->oldpeak) violate this
assumption and can lead to overconfident posterior probabilities.
""")

# ─────────────────────────────────────────────
# Export processed data for all other parts
# ─────────────────────────────────────────────
import joblib

joblib.dump({
    'X_train'     : X_train_sm,
    'X_test'      : X_test_enc,
    'y_train'     : y_train_sm,
    'y_test'      : y_test,
    'X_train_raw' : X_train_raw,
    'X_test_raw'  : X_test_raw,
    'y_train_orig': y_train,
    'scaler'      : scaler,
    'feature_names': list(X_train_enc.columns),
    'df_numeric'  : df[numeric_cols],
    'df_full'     : df,
}, os.path.join(MODELS_DIR, 'preprocessed_data.pkl'))

print("\nPreprocessing complete.")
print(f"  Data   saved -> {os.path.join(MODELS_DIR, 'preprocessed_data.pkl')}")
print(f"  Plot   saved -> {os.path.join(OUTPUTS_DIR, 'pre_correlation_heatmap.png')}")