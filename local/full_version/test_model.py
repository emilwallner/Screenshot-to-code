
# coding: utf-8

# In[1]:


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
from keras.models import model_from_json
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import numpy as np
import os
#get_ipython().magic('matplotlib inline')
#from matplotlib import pyplot as plt


# In[2]:


def load_data(data_dir):
    text = []
    images = []
    all_filenames = os.listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames)[-4:]:
        if filename[-3:] == "npz":
            image = np.load(data_dir+filename)
            images.append(image['features'])
        else:
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            syntax = ' '.join(syntax.split())
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    text = np.array(text)
    return images, text


# In[3]:


num_words = 17
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

tokenizer = Tokenizer(num_words=num_words, filters='', split=" ", lower=False)
tokenizer.fit_on_texts([load_doc('bootstrap.vocab')])


# In[4]:


dir_name = './eval_light/'
train_features, texts = load_data(dir_name)


# In[5]:


#load model and weights 
json_file = open('../../../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../../../weights.h5")
print("Loaded model from disk")


# In[6]:


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[7]:


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    for i in range(117):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = loaded_model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        if word == '<END>':
            break
    return in_text


# In[8]:


def calc_score(real_values, predicted_values):
    total = 000.1
    error = 000.1
    for i in range(len(real_values)):
        for k in range(min([len(predicted_values[i]), len(real_values[i][0])])):
            if predicted_values[i][k] != real_values[i][0][k]:
                error += 1
#                 print(predicted_values[i][k], real_values[i][k])
            total += 1
        if len(predicted_values[i]) != len(real_values[i]):
            error += 1
        print(error, total)
    return 1 - (error / total)


# In[9]:


max_length = 48 
from nltk.translate.bleu_score import sentence_bleu
from numpy import argmax
# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for i in range(len(texts)):
        yhat = generate_desc(model, tokenizer, photos[i], max_length)
        # store actual and predicted
        #print('\n\nReal----> \n\n' + texts[i] + '\n\nPrediction---->\n\n' + yhat)
        actual.append(texts[i].split())
        predicted.append(yhat.split())
    # calculate BLEU score
    #bleu = sentence_bleu(actual, predicted)
 #   print(predicted)
  #  print("actual\n\n\n")
 #   print(actual)
    return actual, predicted

actual, predicted = evaluate_model(loaded_model, texts, train_features, tokenizer, max_length)


# In[102]:


#test = test.replace(",", "")
# print(test)
#test = test.split()
# print(test)
# print(test)
#bootstrap_markup = compiler.compile(test, rendering_function=render_content_with_text)
# print(actual[0])
#print(test)


# In[103]:


from compiler.classes.Utils import *
from compiler.classes.Compiler import *


# In[105]:


#from IPython.core.debugger import Tracer

FILL_WITH_RANDOM_TEXT = True
TEXT_PLACE_HOLDER = "[]"

dsl_path = "compiler/assets/web-dsl-mapping.json"
compiler = Compiler(dsl_path)

#print(test)

def render_content_with_text(key, value):
    if FILL_WITH_RANDOM_TEXT:
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text())
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, Utils.get_random_text(length_text=5, space_number=0))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  Utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
    return value


bootstrap_markup = compiler.compile(actual[0], 'index.html', rendering_function=render_content_with_text)
#compiler.compile(test, rendering_function=render_content_with_text)


# In[101]:


print(bootstrap_markup)


# In[ ]:




