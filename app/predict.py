import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow.keras
from tensorflow.keras import models
from tensorflow.keras import layers
import pathlib
from six.moves.urllib.request import urlopen

labels = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

def predictGenre(url, song):
    data = pd.read_csv('C:/Users/sai/Desktop/data/features_3_sec.csv')
    data = data.drop(['filename'],axis=1)
    data = data.drop(['length'],axis=1)
    data = data.drop(['perceptr_mean'],axis=1)
    data = data.drop(['perceptr_var'],axis=1)
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)

    # normalizing
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    # spliting of dataset into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
                
    history = model.fit(X_train,
                        y_train,
                        epochs=20,
                        batch_size=128)
    epochs = range(1,20)
    
                        
    # calculate accuracy
    test_loss, test_acc = model.evaluate(X_test,y_test)
    print('test_acc: ',test_acc)
    # pd.DataFrame(history.history).plot(figsize=(8,5))
    # plt.show()

    z = io.BytesIO(urlopen(url).read())
    pathlib.Path(('C:/Users/sai/Desktop/data/song.wav')).write_bytes(z.getbuffer())
    y, sr = librosa.load('C:/Users/sai/Desktop/data/song.wav', mono=True, duration=180)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.feature.chroma_cqt(y=y,sr=sr)
    tempo, frames = librosa.beat.beat_track(y=y,sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.var(chroma_stft)} {np.mean(rmse)} {np.var(rmse)} {np.mean(spec_cent)} {np.var(spec_cent)} {np.mean(spec_bw)} {np.var(spec_bw)} {np.mean(rolloff)} {np.var(rolloff)} {np.mean(zcr)} {np.var(zcr)} {np.mean(harmony)} {np.var(harmony)} {tempo}'    
    for e in mfcc:
        to_append += f' {np.mean(e)} {np.var(e)}'

    file = open('C:/Users/sai/Desktop/data/data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    songdata = pd.read_csv('C:/Users/sai/Desktop/data/data.csv')
    scaler = StandardScaler()
    rows = songdata.shape[0]
    inp = np.array(songdata.iloc[rows-1:, :], dtype = float)
    inpT=scaler.fit_transform(inp.T)

    # predictions
    predictions = model.predict(inpT.T)
    maxi=predictions[0][0]
    pos=0
    for i in range(1, 10):
        if maxi<predictions[0][i] and i!=8:
            maxi=predictions[0][i]
            pos=i
    #output = labels[np.argmax(predictions[0])]
    output = labels[pos]
    return output





