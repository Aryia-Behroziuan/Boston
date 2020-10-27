#---Data Seats Homes-Boston In Machine Learning---

#---importing Source---
import sklearn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#---Runing Simple Programming---
boston = load_boston()
bostonDf = pd.DataFrame(boston.data, columns=boston.feature_names)
bostonDf['Price'] = boston.target
bostonDf    

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Prices')
plt.ylabel('Predicted Prices')
plt.show()






