from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

train_data = pd.read_csv(r'/content/train_AIC 2.csv')
X = train_data.drop(columns=['Месяц3', 'Количество позиций', 'y'])
y = train_data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
beta = 0.5
f_beta_05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
beta = 2
f_beta_2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
f1_macro = f1_score(y_test, y_pred, average='macro')

print("F1:", f1_macro)
print("Precision:", precision)
print("Recall:", recall)
print("F0.5:", f_beta_05)
print("F2:", f_beta_2)
