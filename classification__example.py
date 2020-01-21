import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_breast_cancer()

print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)

#---------------------------Splitting the data into training and test sets-------------------------------
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)
N, D = X_train.shape

#--------------------------Scaling the data ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------------------------- TENSORFLOW STUFF ---------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#Alternative you can do:
#model = tf.keras.models.Sequential()
#model.add( tf.keras.layers.Dense(1, input_shape=(D,), activation='sigmoid'))

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#-----------------------------TRAIN THE MODEL ----------------------------------------------------------
r = model.fit(X_train,y_train, validation_data = (X_test,y_test), epochs = 100)

#-----------------------------EVALUATE THE MODEL --------------------------------------------- evaluate() returns loss and accuracy
print("Train Score: ", model.evaluate(X_train,y_train))
print("Test Score: ", model.evaluate(X_test,y_test))

#---------------------PLOTS-------------------------------------------
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

#Accuracy plot
plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()


#----------------------Making predictions --------------------------------
P = model.predict(X_test)
print(P) #They are outputs of the sigmoid function interpreted as probabilities p(y=1|x)

#Round to get the actual predictions
#Note: it has to be flattened since the targets are size (N,) while the predictions are size (N,1)
P = np.round(P).flatten()
print(P)

# ------------------Calculate accuarcy compared to the output ------------------------------
print("Manually calculated accuracy: ", np.mean(P==y_test))
print("Evaluate output: ", model.evaluate(X_test,y_test))


#-------------------------------SAVING AND LOADING A MODEL ------------------------------

model.save('classificationExample.h5')

""" #------------------------------- To load a model ------------------------------
import tensorflow as tf

model = tf.keras.models.load_model('classificationExample.h5')
print(model.layers)
print(model.layers[0].get_weights()) """