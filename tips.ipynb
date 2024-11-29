import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop('target',axis=1)
y_train = train['target'] 
X_train = pd.get_dummies(X_train)
test = pd.get_dummies(test)

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_tr,y_tr)
pred = model.predict(X_val) #roc-auc 만 predict_proba
pred[:10] 
from sklearn.metrics import mean_squared_error
mae = mean_squared_error(y_val, pred)
print(mae)

pred = model.predict(test)
pred[:10]

submit = pd.DataFrame({'pred':pred}) #pred[:,1]
submit.to_csv('result.csv',index=False)
pd.read_csv('result.csv')
