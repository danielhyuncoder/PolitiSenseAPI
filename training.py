import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df=pd.read_csv("ExtractedTweets.csv")
df.dropna(inplace=True)

X=df['Tweet'].values

y=[]
for i in df['Party'].values:
    if i =='Democrat':
        y.append(0)
    else:
        y.append(1)

tokenizer=Tokenizer()
tokenizer.fit_on_texts(X)
sequences=pad_sequences(tokenizer.texts_to_sequences(X), 50)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 64))
model.add(tf.keras.layers.LSTM(32, activation="relu"))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="RMSprop", loss="binary_crossentropy", metrics=['accuracy'])

model.fit(sequences, np.array(y), epochs=4)

# model saving

model.save("politicpolarity.h5")
