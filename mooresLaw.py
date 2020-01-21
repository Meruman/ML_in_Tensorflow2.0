import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv'
tf.keras.utils.get_file('moore.csv', url)
df = pd.read_csv('~/.keras/datasets/moore.csv', header=None).values
X=df[:,0].reshape(-1,1) #Make it a 2D array of size N x D where D=1
y=df[:,1]
#print(X)

#-----------------------Scatter plot of the data ---------------------------------
plt.scatter(X,y)
plt.show()

#-----------------we want a linear model, so we take the log --------------------------
y = np.log(y)
plt.scatter(X,y)
plt.show()

#------------------ Center X so we dont have big numbers -------------------------------
X = X - X.mean()

#----------------- Create tensorflow model---------------------------------------
model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape = (1,)),
        tf.keras.layers.Dense(1)
])

#Stochastic gradient descent SGD has parameters = (learning_rate, momentum)
model.compile(optimizer = tf.keras.optimizers.SGD(0.001,0.9), loss = 'mse')
#model.compile(optimizer = 'adam', loss = 'mse')


#----------------------Learning rate scheduler ------------------------------
def schedule(epoch, lr):
    if epoch >=50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

#-------------------------Train the model -------------------------------
r = model.fit(X, y, epochs = 200, callbacks = [scheduler])

#-----------------------Plot the loss -----------------------
plt.plot(r.history['loss'],label = 'loss')
plt.show()


#------------Get the train weights of the model, get the slope of the line ------------------
print(model.layers) #Note that there are only 1 layer, the input layer doesn't count
print(model.layers[0].get_weights())

# the slope of the line is:

a = model.layers[0].get_weights()[0][0,0]

print("Time to double: ", np.log(2)/a)


#---------------------------Making predictions ----------------------------------------------
#Make sure the line fits our data
Yhat = model.predict(X).flatten()
plt.scatter(X,y)
plt.plot(X,Yhat)
plt.show()

#Manual calculation
#Get the weights
w,b = model.layers[0].get_weights()

#Reshape X because we flattened it again earlier
X = X.reshape(-1,1)

#(N x 1) * (1 x 1) + (1) --> (N x 1)
Yhat2 = (X.dot(w) + b).flatten()

#Don't use == for floating points
print(np.allclose(Yhat,Yhat2))