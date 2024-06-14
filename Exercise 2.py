import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv("shaft_radius.csv")
x_train, x_test, y_train, y_test = train_test_split(df.measurement_index, df.shaft_radius, test_size = 0.20, random_state= 42)