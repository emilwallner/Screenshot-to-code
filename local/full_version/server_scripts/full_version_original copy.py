from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Embedding,GRU,TimeDistributed,RepeatVector, LSTM, concatenate , Input, Reshape
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K 
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model
from tqdm import tqdm
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Dropout, RepeatVector, LSTM, concatenate, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import os



dir_name = './userai/jobs/screenshot2code/train/'
IR2 = InceptionResNetV2(weights='imagenet', include_top=False)

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_data(data_dir):
    text = []
    images = []
    for filename in tqdm(os.listdir(data_dir)):
        if filename[-3:] == "npz":
            image = np.load(data_dir+filename)
            images.append(image['features'])
        else:
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            syntax = ' '.join(syntax.split())
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    return images, text

train_features, texts = load_data(dir_name)



num_words = 17

tokenizer = Tokenizer(num_words=num_words, filters='', split=" ", lower=False)
tokenizer.fit_on_texts(texts)

vocab_size = len(tokenizer.word_index)
train_sequences = tokenizer.texts_to_sequences(texts)
max_sequence = max(len(s) for s in train_sequences)
max_length = 48

def preprocess_data(sequences, features):
    X, y, image_data = list(), list(), list()
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            image_data.append(features[img_no])
            X.append(in_seq[-48:])
            y.append(out_seq)
    return np.array(X), np.array(y), np.array(image_data)

X, y, image_data = preprocess_data(train_sequences, train_features)


image_model = Sequential()
image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(299, 299, 3,)))
image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))
image_model.add(Dropout(0.25))

image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))
image_model.add(Dropout(0.25))

image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))
image_model.add(Dropout(0.25))

image_model.add(Flatten())
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))

image_model.add(RepeatVector(max_length))

visual_input = Input(shape=(299, 299, 3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(max_length,))
language_model = Embedding(vocab_size, 50, input_length=max_length, mask_zero=True)(language_input)
language_model = LSTM(128, return_sequences=True)(language_model)
language_model = LSTM(128, return_sequences=True)(language_model)

decoder = concatenate([encoded_image, language_model])

decoder = LSTM(512, return_sequences=True)(decoder)
decoder = LSTM(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs=[visual_input, language_input], outputs=decoder)

optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


filepath="./userai/jobs/screenshot2code/org-weights-epoch-{epoch:04d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=1)
callbacks_list = [checkpoint]
model.fit([image_data, X], y, batch_size=64, shuffle=False, validation_split=0.1, callbacks=callbacks_list, verbose=1, epochs=1)

# Save model
model_json = model.to_json()
with open("./userai/jobs/screenshot2code/model-org.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./userai/jobs/screenshot2code/org_version_scale_model.h5")

