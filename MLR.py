import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi.csv')

X = data.iloc[:, 0:-1].values
y = data.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2, random_state=0)

reg = LinearRegression()
reg.fit(X_train, y_train)
# print("Model trained! Score: ", rf_reg.score(X_test, y_test),rf_reg.score(X_train, y_train))

pickle.dump(reg, open('taxi_model.pkl','wb'))

model = pickle.load(open('taxi_model.pkl','rb'))
print(model.predict([[80, 1770000, 6000, 85]]))
