import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('train_AIC.csv')
target = 'y'

# Mark missing in "days between" and impute
days_cols = [c for c in df.columns if c.startswith('Дней')]
df[days_cols] = df[days_cols].replace(-1, np.nan)

# Identify features
cat_feats = [c for c in df 
             if df[c].dtype=='int64' 
             and df[c].nunique() < 50 
             and c != target]
num_feats = [c for c in df if c not in cat_feats + [target]]

# Missing-value handling
for col in num_feats:
    df[col + '_miss'] = df[col].isna().astype(int)
    df[col].fillna(df[col].median(), inplace=True)

# Categorical encoding
for col in cat_feats:
    te_map = df.groupby(col)[target].mean()
    df[col + '_te'] = df[col].map(te_map)
    freq_map = df[col].value_counts(normalize=True)
    df[col + '_fe'] = df[col].map(freq_map)

# Interaction features 
sample = df.sample(n=50000, random_state=42)
corr_scores = sample[num_feats].corrwith(sample[target]).abs().sort_values(ascending=False)
top5 = corr_scores.head(5).index.tolist()
for i in range(len(top5)):
    for j in range(i+1, len(top5)):
        f1, f2 = top5[i], top5[j]
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]

# Group aggregations
for cat in ['Поставщик', 'Материал']:
    agg = df.groupby(cat)[num_feats].agg(['mean','std'])
    agg.columns = [f'{cat}_{n}_{stat}' for n,stat in agg.columns]
    df = df.join(agg, on=cat)

# PCA
pca = PCA(n_components=2, random_state=42)
df[['pca_1','pca_2']] = pca.fit_transform(df[num_feats])

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[num_feats])

print("New feature matrix size:", df.shape)
print("Sample of engineered features:")
print(df.filter(regex=r'(_miss|_te|_fe|_x_|^Поставщик_|^Материал_|pca_|cluster)').head())
