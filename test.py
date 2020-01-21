""" import tensorflow as tf
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
tf.keras.utils.get_file('auto-mpg.data', url)
df = pd.read_csv('~/.keras/datasets/auto-mpg.data', header=None, delim_whitespace=True)
print(df.head()) """


#------------------------------- To load a model ------------------------------
import tensorflow as tf

model = tf.keras.models.load_model('classificationExample.h5')
print(model.layers)
print(model.layers[0].get_weights())


#Rectifier Linear unit, exponential linear unit, leaky ReLU, softplus