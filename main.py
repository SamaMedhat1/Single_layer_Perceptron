import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot

data = pd.read_csv('penguins.csv')

# find important columns name which contain  numeric values
numbers_cols = data.select_dtypes(include=np.number).columns.to_list()

# find important columns name which contain nun numeric values & convert it's type to string
non_integer_cols = data.select_dtypes(include=['object']).columns.to_list()
data[non_integer_cols] = data[non_integer_cols].astype('string')

# encode species column
label_encoders = []
label_encoder = preprocessing.LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])
label_encoders.append(label_encoder)

# split data based on specie
Adelie = data.iloc[0:50, :]
Gentoo = data.iloc[50: 100, :]
Chinstrap = data.iloc[100: 150, :]

nan_val_in_Adelie = {}
nan_val_in_Gentoo = {}
nan_val_in_Chinstrap = {}

# find values for 'nan' with median in integer cols & with most repeated value in 'gender' col.
# for integer col
for col in numbers_cols:
    nan_val_in_Adelie[col] = Adelie[col].median()
    nan_val_in_Gentoo[col] = Gentoo[col].median()
    nan_val_in_Chinstrap[col] = Chinstrap[col].median()

# for gender
nan_val_in_Adelie['gender'] = Adelie['gender'].mode()[0]
nan_val_in_Gentoo['gender'] = Gentoo['gender'].mode()[0]
nan_val_in_Chinstrap['gender'] = Chinstrap['gender'].mode()[0]

# replace nan
# in Adelie
Adelie = Adelie.fillna(value=nan_val_in_Adelie)
# in Gentoo
Gentoo = Gentoo.fillna(value=nan_val_in_Gentoo)
# in Chinstrap
Chinstrap = Chinstrap.fillna(value=nan_val_in_Chinstrap)

# Encoding gender column
genders = ['male', 'female']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(genders)
Adelie[Adelie.columns[4]] = label_encoder.transform(Adelie['gender'])
Gentoo[Gentoo.columns[4]] = label_encoder.transform(Gentoo['gender'])
Chinstrap[Chinstrap.columns[4]] = label_encoder.transform(Chinstrap['gender'])
label_encoders.append(label_encoder)

print(Gentoo)
