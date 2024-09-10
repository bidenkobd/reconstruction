import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf                         ### works with v2.11.0
from tensorflow.keras import backend as K
tf.config.list_physical_devices('GPU')
from UNET import UNET3D                         ### reference the original paper: https://doi.org/10.48550/arXiv.2405.05598 if you use this module
import numpy as np
from tqdm import tqdm

BoxSize = 1000.
kF = 2*np.pi/BoxSize
grid = 256

### loading training and validation data
### all data should have mean=0 and stddev=1
### the array shape is (sample size,256,256,256,1)

# input field
X_train = np.load('./tb_train.npy',allow_pickle = True)
X_val = np.load('./tb_val.npy',allow_pickle = True)

# output field (ts,xh,delta)
y_train = np.load('./ts_train.npy',allow_pickle = True)
y_val = np.load('./ts_val.npy',allow_pickle = True)


K.clear_session()
model = UNET3D(256, n_base_filters=8,depth=5,loss_function=tf.keras.losses.mse,initial_learning_rate=1e-5)
model.summary()

def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * 0.99

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = "./ts"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
# model.load_weights("./ts_pretrained")   ### use to continue from the previous checkpoint

model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), shuffle=True,batch_size=1, epochs=3000,validation_freq=1,callbacks=[model_checkpoint_callback])