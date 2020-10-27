#---Anal Homes-Bostons in Machine Learning---

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
print(bostonDf)   