"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

from keras.layers import Embedding

from keras.layers import CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import datetime
import matplotlib.pyplot as plt
from random import sample
import random
import keras
import string
from keras.preprocessing.text import text_to_word_sequence
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#define batch size
batch_size_num=128
#define number of units
num_input_words = 100
#define sequence length
seq_length=30
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#import file
filename = "C:/Documents/AI/RNN ass/abcnews-date-text.csv"
#transfrom file into a dataframe
df = pd.read_csv(filename, 
            header=0, 
            names=['publish_date', 'headline_text'])
#random.seed(1)
df=df.sample(n=10000,replace=False,random_state=1)
#drop the colomn of time stamp
df=df.drop(columns=['publish_date'])
#choose a subset of dataset
df.to_csv("C:/Documents/AI/RNN ass/cleanedup-news-file-test.csv",
          index=False,header=False)

#import data after preprocessing
filename="C:/Documents/AI/RNN ass/cleanedup-news-file-test.csv"
#read data into raw text
raw_text = open(filename, encoding="utf8").read()
#lowercase
raw_text = raw_text.lower()
#transfrom punctuations into integer
dropPunctuation = str.maketrans("", "", string.punctuation)
raw_text = raw_text.translate(dropPunctuation)
#define start time
start_time1 = datetime.datetime.now()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# create mapping of unique words to integers
lines = text_to_word_sequence(raw_text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
encoded = tokenizer.texts_to_sequences(lines)
encoded_sequence=[]
for list in encoded:
    encoded_sequence.append(list[0])
sequences=[]
#create sequence for training
for i in range(0, len(encoded_sequence)-seq_length-1):
	sequence = encoded_sequence[i:i+seq_length+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('vocab_size=',vocab_size)
# separate into input and output
sequences = array(sequences)
#define x and y
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Define Model 1 - CuDNN version Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

epoch_num =50

# define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(num_input_words, return_sequences=True))
model.add(LSTM(num_input_words))
model.add(Dense(num_input_words, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Train Model 1 Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath="C:/Documents/AI/RNN ass/learn-by-wordModel-2-LSTM-weights-improvement-10-7.1778-dropout.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, period = 10, mode='min')
callbacks_list = [checkpoint]

# fit the model
history1 = model.fit(X, y, epochs=epoch_num, batch_size=batch_size_num, callbacks=callbacks_list)
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
stop_time1 = datetime.datetime.now()
#plot model loss graph
print("Model 1 Summary")
print("Batch Size:",batch_size_num,"\nNumber of Epochs:",epoch_num)
model.summary()
print("Last loss score:",history1.history['loss'][-1] )
print ("Time required for training:",stop_time1 - start_time1)

# summarize history for loss
plt.plot(history1.history['loss'])
plt.title('model 1 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper right')
plt.show()

Write by word code:
import pandas as pd

import string
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
epoch_num =50
num_words_to_generate = 8
seq_length=30
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

filename = "C:/Documents/AI/RNN ass/abcnews-date-text.csv"
df = pd.read_csv(filename, 
            header=0, 
            names=['publish_date', 'headline_text'])
#random.seed(1)
df=df.sample(n=100,replace=False,random_state=1)
#drop time column
df=df.drop(columns=['publish_date'])
#save file as csv
df.to_csv("C:/Documents/AI/RNN ass/cleanedup-news-file-test.csv",
          index=False,header=False)
#load csv file
filename="C:/Documents/AI/RNN ass/cleanedup-news-file-test.csv"
#raw_text
raw_text = open(filename, encoding="utf8").read()
#lowercase
raw_text = raw_text.lower()
#drop punctuations
dropPunctuation = str.maketrans("", "", string.punctuation)
raw_text = raw_text.translate(dropPunctuation)
#spit line
lines = raw_text.split('\n')


# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = []
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# load the model
model = load_model('units100-learn-by-wordModel-2-LSTM-weights-improvement-50-3.6828-dropout.hdf5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# generate new text
# select a seed text
for i in range(20):
    seed_text = lines[randint(0,len(lines))]
    print('seed=',seed_text + '\n')
    generated = generate_seq(model, tokenizer, seq_length, seed_text, 
                             num_words_to_generate)
    print('generated=',generated)

