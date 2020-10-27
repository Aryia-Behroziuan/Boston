#---Path MSE Datasete Boston---

#---importing Source---
import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#---Runing Simple Programming---
boston = load_boston()
bostonDf = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDf['Price'] = boston.target
bostonDf    

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#---Model---
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.show()

mse = mean_squared_error(y_test, y_pred)
print(mse)

#new_x = boston.data[:,[1, 2]]
#new_y = boston.target

#new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size=0.3, random_state=42)

#new_reg = LinearRegression()
#new_reg.fit(new_x_train, new_y_train)
#new_y_predict = new_reg.predict(new_x_test)

#new_mse = mean_squared_error(new_y_test, new_y_predict)
#print(new_mse)

reg1 = LinearRegression()
cv_scores = cross_val_score(reg1, x, y, cv=5)
