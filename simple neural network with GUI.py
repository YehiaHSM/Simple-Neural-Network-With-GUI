import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter.ttk import *

# GUI setup
master = Tk()
master.geometry("400x250")

# Input fields and labels
e1 = Entry(master)
e1.place(x=130, y=25)
e1.focus_set()

e2 = Entry(master)
e2.place(x=130, y=50)
e2.focus_set()

variable = StringVar(master)
variable.set("Choose")  # default value

w = OptionMenu(master, variable, "Sigmoid", "tanh")
w.pack()
w.place(x=130, y=75)

e4 = Entry(master)
e4.place(x=130, y=100)
e4.focus_set()

e5 = Entry(master)
e5.place(x=130, y=125)
e5.focus_set()

e6 = Entry(master)
e6.place(x=130, y=150)
e6.focus_set()

var = StringVar()
var.set("Layers")
label = Label(master, textvariable=var, relief=RAISED)
label.pack()

var2 = StringVar()
var2.set("Neurons")
label2 = Label(master, textvariable=var2, relief=RAISED)
label2.pack()

var3 = StringVar()
var3.set("Activation Function")
label3 = Label(master, textvariable=var3, relief=RAISED)
label3.pack()

var4 = StringVar()
var4.set("eta")
label4 = Label(master, textvariable=var4, relief=RAISED)
label4.pack()

var5 = StringVar()
var5.set("epochs")
label5 = Label(master, textvariable=var5, relief=RAISED)
label5.pack()

var6 = StringVar()
var6.set("bias")
label6 = Label(master, textvariable=var6, relief=RAISED)
label6.pack()

label.place(x=20, y=25)
label2.place(x=20, y=50)
label3.place(x=20, y=75)
label4.place(x=20, y=100)
label5.place(x=20, y=125)
label6.place(x=20, y=150)

# Checkbox for bias
Ba = IntVar()
b = Checkbutton(master, text="Bias", variable=Ba)
b.pack()
b.place(x=270, y=150)

O2 = [0, 0, 0, 0, 0, 0, 0]

def Run1():
    i = 0
    O2[i] = e1.get()
    i += 1
    O2[i] = e2.get()
    i += 1
    O2[i] = variable.get()
    i += 1
    O2[i] = e4.get()
    i += 1
    O2[i] = e5.get()
    i += 1
    O2[i] = e6.get()
    i += 1
    O2[i] = Ba.get()
    print(O2)

B = Button(master, text="Run", command=Run1)
B.pack()
B.place(x=150, y=200)
mainloop()

# Retrieve parameters from GUI
feature1, feature2, activation_function, alpha, epochs, bias, _ = O2

# Convert feature2 to list of neurons
list12 = list(map(int, feature2.split(',')))
print("Layers:", list12)

# Activation Functions
def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

# Layer class definition
class Layer:
    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid)
    }

    def __init__(self, inputs, neurons, activation):
        self.W = np.random.randn(neurons, inputs)
        self.b = np.zeros((neurons, 1)) if bias else None
        self.act, self.d_act = self.activationFunctions.get(activation)

    def feedforward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev.T).T
        if self.b is not None:
            self.Z += self.b.T
        self.A = self.act(self.Z)
        return self.A

# Load and preprocess data
df = pd.read_csv('IrisData.csv')
df_one_hot = pd.get_dummies(df)
train_x, test_x, train_y, test_y = train_test_split(df_one_hot.iloc[:, :4], df_one_hot.iloc[:, 4:], test_size=0.4, random_state=32)

x_train = np.array(train_x)
y_train = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

# Initialize layers
layers = [Layer(4, list12[0], activation_function)]
for i in range(1, len(list12)):
    layers.append(Layer(list12[i-1], list12[i], activation_function))

# Training
for epoch in range(int(epochs)):
    for i in range(x_train.shape[0]):
        A = x_train[i, :].reshape(1, -1)
        for layer in layers:
            A = layer.feedforward(A)

# Predictions
y_hat = []
for i in range(test_x.shape[0]):
    A = test_x[i, :].reshape(1, -1)
    for layer in layers:
        A = layer.feedforward(A)
    y_hat.append(A)

print(y_hat)
