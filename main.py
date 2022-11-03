import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt, pyplot
from tkinter import *
from tkinter.ttk import *

# GUI
form = Tk()
classes = ['Adelie', 'Gentoo', 'Chinstrap']
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g']
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


def user_inputs():
    selectedClass1 = data1.get()
    selectedClass2 = data2.get()
    selectedFeature1 = data3.get()
    selectedFeature2 = data4.get()
    lR = var1.get()
    epochNum = var2.get()
    use_bias = True
    if radio_var.get() == 1:
        use_bias = True
    elif radio_var.get() == 2:
        use_bias = False

    return selectedClass1, selectedClass2, selectedFeature1, selectedFeature2, lR, epochNum, use_bias


def initialize_Model_Dfs():
    selectedClass1, selectedClass2, selectedFeature1, selectedFeature2, lR, epochNum, use_bias = user_inputs()

    # create train & test data based on user selection
    # 1) select species
    train_frames = []
    test_frames = []
    if selectedClass1 == 'Adelie':
        train_frames.append(Adelie_train)
        test_frames.append(Adelie_test)
    elif selectedClass1 == 'Gentoo':
        train_frames.append(Gentoo_train)
        test_frames.append(Gentoo_test)
    else:
        train_frames.append(Chinstrap_train)
        test_frames.append(Chinstrap_test)

    if selectedClass2 == 'Adelie':
        train_frames.append(Adelie_train)
        test_frames.append(Adelie_test)
    elif selectedClass2 == 'Gentoo':
        train_frames.append(Gentoo_train)
        test_frames.append(Gentoo_test)
    else:
        train_frames.append(Chinstrap_train)
        test_frames.append(Chinstrap_test)

    train_data = pd.concat(train_frames).reset_index(drop=True)
    test_data = pd.concat(test_frames).reset_index(drop=True)

    # 2) keep only selected features
    train_data = train_data[['species', selectedFeature1, selectedFeature2]]
    test_data = test_data[['species', selectedFeature1, selectedFeature2]]

    # encode species column
    label_encoder = preprocessing.LabelEncoder()
    train_data['species'] = label_encoder.fit_transform(train_data['species'])
    test_data['species'] = label_encoder.transform(test_data['species'])

    # data shuffling
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    # separate labels
    train_labels = train_data.pop('species')
    test_labels = test_data.pop('species')

    # weight & bias

    if use_bias:
        weights = np.random.rand(3)
    else:
        weights = np.random.rand(2)

    return train_data, train_labels, test_data, test_labels, weights, epochNum, lR


def run():
    train_data, train_labels, test_data, test_labels, weights, epochNum, lr = initialize_Model_Dfs()
    run_single_layer(train_data, train_labels, weights, epochNum, lr)


def run_single_layer(train_data, train_label, weights, epochNum, lr):
    trainData = train_data.to_numpy()
    trainLabel = train_label
    transpose_weight = weights.transpose()
    bias = 1
    row_num = 0
    while epochNum:
        for row in trainData:
            if len(weights) > 2:
                row = np.append(row, bias)
            net = np.dot(row, transpose_weight)
            predictedValue = np.sign(net)
            error = trainLabel[row_num] - predictedValue
            if error != 0:
                update_weight(transpose_weight, lr, row, error)
            row += 1


def update_weight(weight_matrix, l_rate, row, error_value):
    for index in range(len(weight_matrix)):
        weight_matrix[index] = weight_matrix[index] + l_rate * error_value * row[index]


def create_label():
    class_label = Label(form, textvariable=label1)
    label1.set("Select the two species")
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
    btn = Button(form, text="Run", command=run)
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


def data_preprocessing():
    dataSet = pd.read_csv('penguins.csv')

    # find important columns name which contain  numeric values
    numbers_cols = dataSet.select_dtypes(include=np.number).columns.to_list()

    # find important columns name which contain nun numeric values & convert it's type to string
    non_integer_cols = dataSet.select_dtypes(include=['object']).columns.to_list()
    dataSet[non_integer_cols] = dataSet[non_integer_cols].astype('string')

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

    # Encoding gender column
    genders = ['male', 'female']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(genders)
    adelie[adelie.columns[4]] = label_encoder.transform(adelie['gender'])
    gentoo[gentoo.columns[4]] = label_encoder.transform(gentoo['gender'])
    chinstrap[chinstrap.columns[4]] = label_encoder.transform(chinstrap['gender'])

    # dataSet shuffling
    adelie = adelie.sample(frac=1).reset_index(drop=True)
    gentoo = gentoo.sample(frac=1).reset_index(drop=True)
    chinstrap = chinstrap.sample(frac=1).reset_index(drop=True)

    # split dataSet into train dataSet and test dataSet
    Adelie_train = adelie.iloc[:30, :]
    Adelie_test = adelie.iloc[30:, :].reset_index(drop=True)
    Gentoo_train = gentoo.iloc[:30, :]
    Gentoo_test = gentoo.iloc[30:, :].reset_index(drop=True)
    Chinstrap_train = chinstrap.iloc[:30, :]
    Chinstrap_test = chinstrap.iloc[30:, :].reset_index(drop=True)

    return Adelie_train, Adelie_test, Gentoo_train, Gentoo_test, Chinstrap_train, Chinstrap_test


Adelie_train, Adelie_test, Gentoo_train, Gentoo_test, Chinstrap_train, Chinstrap_test = data_preprocessing()

gui()
def test (TestLabel,test_data,weights):
    testData = test_data.to_numpy()
    transpose_weight = weights.transpose()
    testLabel = TestLabel
    row_num = 0
    x0 = 1
    for row in testData:
        if len(weights) > 2:
            row = np.append(row, x0)
        net = np.dot(row, transpose_weight)
        predictedValue = np.sign(net)
        error = testLabel[row_num] - predictedValue
        if error == 0:
            score = score+1
    accuracy = (score/testData)*100
    print("accuracy:", accuracy, "and the score: ", score)

    return accuracy
def testSample (weight,SampleX):
    SampleX = 10
    transpose_weight = weight.transpose()
    net = np.dot(SampleX, transpose_weight)
    predictedValue_y = np.sign(net)
    print("the ClassID :", predictedValue_y)
    return 0


# bias = 1
# rowNum = 0
# while epochnum:
#     for index in range(len(train_data)):
#         i = 0
#         net = 0
#         for i in range(3):
#             net += train_data[*weights[i]
#             i+=1
#         if use_bias:
#            net += bias * weights[i]
#
#     rowNum += 1
#     epochnum -= 1;
