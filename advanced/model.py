import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('train_AIC.csv') 
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Scale_pos_weight for imbalance
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

# DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

# Parameters
params = {
    'objective':         'binary:logistic',
    'eval_metric':       'aucpr',           # PR AUC for imbalance
    'eta':               0.05,              # learning rate
    'max_depth':         6,
    'subsample':         0.8,
    'colsample_bytree':  0.8,
    'gamma':             1.0,
    'scale_pos_weight':  scale_pos_weight,
    'tree_method':       'hist',
    'seed':              42
}

evallist = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=50,
    verbose_eval=10
)

y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)
print(f"Test F1 Score: {f1_score(y_test, y_pred):.4f}")
