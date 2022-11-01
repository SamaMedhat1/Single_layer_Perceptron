import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot

data = pd.read_csv('penguins.csv')

numbers_cols = data.select_dtypes(include= np.number).columns.to_list()


