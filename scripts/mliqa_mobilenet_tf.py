############### XceptionNet
import logging
import os
import warnings
import time
# import matplotlib.pyplot as plt
# import matplotlib.style as style
import numpy as np
import pandas as pd
import ast
# import seaborn as sns
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import json
from scipy.stats import spearmanr
from datetime import datetime
from keras.preprocessing import image
from PIL import Image
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.calibration import calibration_curve
from tensorflow.keras import layers
from tensorflow.keras import models
from keras import backend as K
import sys
import pickle

model_name = sys.argv[1]

warnings.filterwarnings('ignore')
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# with open('quality.json') as json_file:
#     data = json.load(json_file)


# X_train = data['train']['image']
# X_val = data['val']['image']

df_train = pd.read_csv('train_qual_dist.csv')
df_val = pd.read_csv('val_qual_dist.csv')
df_test = pd.read_csv('test_qual_dist.csv')

X_train = df_train['image'].tolist()
X_val = df_val['image'].tolist()
X_test = df_test['image'].tolist()

X_train = ['../vizwiz/train/'+f+'.jpg' for f in X_train]
X_val = ['../vizwiz/val/'+f+'.jpg' for f in X_val]
X_test = ['../vizwiz/test/'+f+'.jpg' for f in X_test]

train_y_qual = df_train['qual_mos'].tolist()
val_y_qual = df_val['qual_mos'].tolist()
test_y_qual = df_test['qual_mos'].tolist()

df_train['dist_prob'] = df_train['dist_prob'].apply(ast.literal_eval)
train_y_flaw = []
for i in range(len(X_train)):
  y = df_train['dist_prob'][i]
  train_y_flaw.append(y)

df_val['dist_prob'] = df_val['dist_prob'].apply(ast.literal_eval)
val_y_flaw = []
for i in range(len(X_val)):
  y = df_val['dist_prob'][i]
  val_y_flaw.append(y)

df_test['dist_prob'] = df_test['dist_prob'].apply(ast.literal_eval)
test_y_flaw = []
for i in range(len(X_test)):
  y = df_test['dist_prob'][i]
  test_y_flaw.append(y)

IMG_SIZE = 448 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model

def parse_function(filename, label):
    # Read an image from a file
    image_string = tf.io.read_file(filename)
    # Decode it into a dense vector
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    # Resize it to fixed shape
    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_normalized = image_resized / 255.0
    return image_normalized, label


BATCH_SIZE = 64 # Big enough to measure an F1-score
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
SHUFFLE_BUFFER_SIZE = 128 # Shuffle the training data by a chunck of 1024 observations


def create_dataset(filenames, label_1, label_2, is_training):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, {'output_1':label_1,'output_2':label_2}))
    # Parse and preprocess observations in parallel
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    if is_training == True:
        # This is a small dataset, only load it once, and keep it in memory.
        dataset = dataset.cache()
        # Shuffle the data each buffer size
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

train_ds = create_dataset(X_train, train_y_flaw, train_y_qual, is_training=True)
val_ds = create_dataset(X_val, val_y_flaw, val_y_qual, is_training=False)
test_ds = create_dataset(X_test, test_y_flaw, test_y_qual, is_training=False)


# feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
# feature_extractor_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
# feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS),
                                               include_top=False,
                                               weights='imagenet')
# # Fine-tune from this layer onwards
# fine_tune_at = 125
#
# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#   layer.trainable =  False

base_model.trainable = False

# feature_extractor_layer.trainable = False

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def srcc(y_true, y_pred):
     return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )

LR = 1e-5 # Keep it small when transfer learning
EPOCHS = int(sys.argv[2])

def build_model():
    # Define model layers.
    input_layer = layers.Input(shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
    x = base_model(input_layer, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu', name='hidden_layer1')(x)
    x = layers.Dropout(0.2)(x)
    x1 = layers.Dense(32, activation='relu', name='hidden_layer2')(x)
    x1 = layers.Dropout(0.2)(x1)
    output_2 = layers.Dense(1, name='output_2')(x1)

    x2 = layers.Dense(32, activation='relu', name='hidden_layer3')(x)
    x2 = layers.Dropout(0.2)(x2)
    output_1 = layers.Dense(7, name='output_1')(x2)

    # x1 = layers.Dense(256, activation='relu', name='hidden_layer_1')(x)
    # output_1 = layers.Dense(3, activation='sigmoid', name='output_1')(x)
    # x2 = layers.Dense(32, activation='relu', name='hidden_layer_2')(x)
    # output_2 = layers.Dense(1, name='output_2')(x2)

    model = models.Model(inputs=input_layer, outputs=[output_1, output_2])

    return model

model=build_model()
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'output_1': 'mse', 'output_2': 'mse'},
 metrics={'output_1':'mse',
          'output_2':srcc})

lr = float(sys.argv[3])
def scheduler(epoch):
    if epoch < 5:
        return lr
    else:
        return lr*tf.math.exp(-0.1)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
checkpoint_filepath = "./models/"+model_name+'_checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

start = time.time()
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    callbacks=[callback,model_checkpoint_callback],
                    validation_data=val_ds)
print('\nTraining took {}'.format(time.time()-start))

export_path = "./models/"+model_name+".h5"
model.save(export_path)
print("Model was exported in this path: '{}'".format(export_path))

# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history
history_path = 'fit_history_'+model_name+'.pkl'
with open(history_path, 'wb') as f:
    pickle.dump(history_dict, f, pickle.HIGHEST_PROTOCOL)


y_pred = model.predict(test_ds)
with open('test_qual_'+model_name+'.npy', 'wb') as f:
    np.save(f, y_pred[1])
with open('test_dist_'+model_name+'.npy', 'wb') as f:
    np.save(f, y_pred[0])
y_qual_pred = []
for i in range(len(y_pred[1])):
    y_qual_pred.append(y_pred[1][i][0])

# print(val_y_qual)
# print(y_qual_pred)

print('SRCC: ', spearmanr(test_y_qual, y_qual_pred))
