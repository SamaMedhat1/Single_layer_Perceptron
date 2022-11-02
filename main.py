import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot
from tkinter import *
from tkinter.ttk import *

# GUI
form = Tk()
classes = ['Adelie', 'Gentoo', 'Chinstrap']
features = ['body mass', 'flipper length', 'bill depth', 'bill length', 'gender']
data1 = StringVar()
data2 = StringVar()
data3 = StringVar()
data4 = StringVar()
label1 = StringVar()
label2 = StringVar()
label3 = StringVar()
label4 = StringVar()
label5 = StringVar()
var1 = DoubleVar()
var2 = IntVar()
radio_var = IntVar()
class1 = Combobox(form, width=20, textvariable=data1)
class2 = Combobox(form, width=20, textvariable=data2)
feature1 = Combobox(form, width=20, textvariable=data3)
feature2 = Combobox(form, width=20, textvariable=data4)


def run_single_layer():
    



def create_label():
    class_label = Label(form, textvariable=label1)
    label1.set("Select the two features")
    class_label.place(x=20, y=30)

    feature_label = Label(form, textvariable=label2)
    label2.set("Select the two features")
    feature_label.place(x=20, y=120)

    lr_label = Label(form, textvariable=label3)
    label3.set("learning rate")
    lr_label.place(x=20, y=220)

    epoch_label = Label(form, textvariable=label5)
    label5.set("epochs number")
    epoch_label.place(x=250, y=220)


def create_radio():
    r1 = Radiobutton(form, text="bias", width=120, variable=radio_var, value=1)
    r1.pack(anchor=W)
    r1.place(x=120, y=290)

    r2 = Radiobutton(form, text="no bias", width=120, variable=radio_var, value=2)
    r2.pack(anchor=W)
    r2.place(x=300, y=290)


def create_button():
    btn = Button(form, text="Run", command = run_single_layer)
    btn.place(x=190, y=350)


def create_spinbox():

    spin1 = Spinbox(form, from_=0, to=1, increment=0.1, width=5, textvariable=var1)
    spin1.place(x=120, y=220)

    spin2 = Spinbox(form, from_=1, to=100, width=5, textvariable=var2)
    spin2.place(x=350, y=220)


def create_combo():
    class1['values'] = classes
    class1.grid(column=1, row=3)
    class1.place(x=20, y=60)
    class1.current()
    data1.trace('w', update)

    class2['values'] = classes
    class2.grid(column=1, row=2)
    class2.place(x=260, y=60)
    class2.current()

    feature1['values'] = features
    feature1.grid(column=1, row=5)
    feature1.place(x=20, y=150)
    feature1.current()
    data3.trace('w', update2)

    feature2['values'] = features
    feature2.grid(column=1, row=4)
    feature2.place(x=260, y=150)
    feature2.current()


def update(*args):
    data2.set('')
    class2["values"] = [x for x in classes if x != data1.get()]


def update2(*args):
    data4.set('')
    feature2["values"] = [x for x in features if x != data3.get()]


def gui():
    form.geometry("450x450")
    form.title("Form")
    create_label()
    create_combo()
    create_spinbox()
    create_radio()
    create_button()
    form.mainloop()


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

# correct dfs index
# Gentoo.set_index(pd.Index(range(50)), inplace=True)
# Chinstrap.set_index(pd.Index(range(50)), inplace=True)

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

# data shuffling
Adelie = Adelie.sample(frac=1).reset_index(drop=True)
Gentoo = Gentoo.sample(frac=1).reset_index(drop=True)
Chinstrap = Chinstrap.sample(frac=1).reset_index(drop=True)

# split data into train data and test data
Adelie_train = Adelie.iloc[:30, :]
Adelie_test = Adelie.iloc[30:, :].reset_index(drop=True)
Gentoo_train = Gentoo.iloc[:30, :]
Gentoo_test = Gentoo.iloc[30:, :].reset_index(drop=True)
Chinstrap_train = Chinstrap.iloc[:30, :]
Chinstrap_test = Chinstrap.iloc[30:, :].reset_index(drop=True)

gui()
selectedClass1 = data1.get()
selectedClass2 = data2.get()
selectedFeature1 = data3.get()
selectedFeature2 = data4.get()
lR = var1.get()
epochNum = var2.get()
bias = ''
if radio_var.get() == 1:
    bias = True
elif radio_var.get() == 2:
    bias = False

# print(selectedClass1)
# print(selectedClass2)
# print(selectedFeature1)
# print(selectedFeature2)
# print(lR)
# print(epochNum)
# print(bias)
# print(Gentoo_test)
