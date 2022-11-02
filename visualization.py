import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def preprocessing():
    dataSet = pd.read_csv('penguins.csv')
    # find important columns name which contain  numeric values
    numbers_cols = dataSet.select_dtypes(include=np.number).columns.to_list()

    # split dataSet based on specie
    adelie = dataSet.iloc[0:50, :]
    gentoo = dataSet.iloc[50: 100, :]
    chinstrap = dataSet.iloc[100: 150, :]

    nan_val_in_Adelie = {}
    nan_val_in_Gentoo = {}
    nan_val_in_Chinstrap = {}

    # find values for 'nan' with median in integer cols & with most repeated value in 'gender' col.
    # for integer col
    for col in numbers_cols:
        nan_val_in_Adelie[col] = adelie[col].median()
        nan_val_in_Gentoo[col] = gentoo[col].median()
        nan_val_in_Chinstrap[col] = chinstrap[col].median()

    # for gender
    nan_val_in_Adelie['gender'] = adelie['gender'].mode()[0]
    nan_val_in_Gentoo['gender'] = gentoo['gender'].mode()[0]
    nan_val_in_Chinstrap['gender'] = chinstrap['gender'].mode()[0]

    # replace nan
    # in adelie
    adelie = adelie.fillna(value=nan_val_in_Adelie)
    # in gentoo
    gentoo = gentoo.fillna(value=nan_val_in_Gentoo)
    # in Chinstrap
    chinstrap = chinstrap.fillna(value=nan_val_in_Chinstrap)

    # data = pd.concat([adelie, gentoo, chinstrap]).reset_index(drop=True)
    return adelie, gentoo, chinstrap


adelie, gentoo, chinstrap = preprocessing()
columnNames = adelie.columns


for firstFeatureIdx in range(1, 6):
    for secondFeatureIdx in range(firstFeatureIdx + 1, 6):
        firstFeature = columnNames[firstFeatureIdx]
        secondFeature = columnNames[secondFeatureIdx]
        figureName = str(firstFeature) + ' & ' + str(secondFeature)
        plt.figure(figureName)
        plt.scatter(adelie[firstFeature], adelie[secondFeature])
        plt.scatter(gentoo[firstFeature], gentoo[secondFeature])
        plt.scatter(chinstrap[firstFeature], chinstrap[secondFeature])
        plt.xlabel(firstFeature)
        plt.ylabel(secondFeature)
        plt.show()
        # plt.savefig(figureName)
