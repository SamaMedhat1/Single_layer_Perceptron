import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot

data = pd.read_csv('penguins.csv')

# find important columns name which contain  numeric values
numbers_cols = data.select_dtypes(include= np.number).columns.to_list()

# find important columns name which contain nun numeric values & convert it's type to string
non_integer_cols = data.select_dtypes(include=['object']).columns.to_list()
data[non_integer_cols] = data[non_integer_cols].astype('string')

# encode species column
label_encoders = []
label_encoder = preprocessing.LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])
label_encoders.append(label_encoder)


