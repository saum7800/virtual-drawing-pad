import tensorflow as tf
import h5py
print(tf.__version__)
new_model = tf.keras.models.load_model('myEmnistModel.h5')

