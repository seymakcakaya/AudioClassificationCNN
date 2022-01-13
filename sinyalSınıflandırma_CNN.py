#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install librosa


# In[2]:


pip install tensorflow


# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
import tensorflow as tf


# In[2]:



#ses dosyası dijitale çevrildi


# In[3]:



audio_file_path='C:/Users/user/Desktop/UrbanSound8K/17973-2-0-32.wav'

librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)


# In[4]:



audio_file_path='C:/Users/user/Desktop/UrbanSound8K/17973-2-0-32.wav'
librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
print(librosa_audio_data)


# In[5]:



#librosa kütüphnesi stereo yu mono hale getirip okuyor. 2 farklı ses sinyalini mono hale getiriyo 
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
plt.show()


# Actually our sample is stereo, you can see it using scipy:

# In[6]:


# stereo okuyor bunu kullanmıyoruz çnkü mfcc özelliği yok
from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(audio_file_path)


# In[7]:


from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(audio_file_path)
wave_audio


# In[8]:


#  2 channels 
plt.figure(figsize=(12, 4))
plt.plot(wave_audio)
plt.show()


# In[9]:


mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)   #n_mfcc: number of MFCCs to return - kaç öznitelik çıkartılacak 
print(mfccs.shape)


# In[10]:


mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40) 
mfccs


# In[11]:


audio_dataset_path='C:/Users/user/Desktop/UrbanSound8K/audio/'
metadata=pd.read_csv('C:/Users/user/Desktop/UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()


# In[12]:


#np.mean(mfccs_features.T,axis=0) ile scaled işlemi yapıyoruz ve scaled edilmiş değerleri dönderiyoruz.
#her bir ses için öz nitelik çıkartacak
def features_extractor(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


# In[13]:


#  dosyaların içini hepsini okuyor ve arraya atma işlemi gerçekleştiriliyor
# MFCC
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[14]:


#liste değil de dataframe istiyoruz dataframe e çeviriyoruz.
#feature ce class sütunlarını kullaranak pandas data frame oluştur diyoruz
#hers ses dosaysının featıre ve sınıflarını oluşturduk
extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[15]:


X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())#sonuç


# In[16]:


X.shape


# In[17]:


X


# In[18]:


y


# In[19]:


y.shape


# In[20]:


# 1 0 0 0 0 0 0 0 0 0 => air_conditioner
# 0 1 0 0 0 0 0 0 0 0 => car_horn
# 0 0 1 0 0 0 0 0 0 0 => children_playing
# 0 0 0 1 0 0 0 0 0 0 => dog_bark
# ...
# 0 0 0 0 0 0 0 0 0 1 => street_music

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[21]:


y


# In[22]:


y[0]#köpek havlaması


# In[23]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[24]:


X_train


# In[25]:


y


# In[26]:


X_train.shape


# In[27]:


X_test.shape


# In[28]:


y_train.shape


# In[29]:


y_test.shape


# In[30]:


num_labels = 10


# In[31]:


# CNN model

model=Sequential()
# 1. hidden layer
model.add(Dense(125,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 2. hidden layer
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# 3. hidden layer
model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# çıkış
model.add(Dense(num_labels))#10 bitlik output verecek
model.add(Activation('softmax'))


# In[32]:


model.summary()


# In[33]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[34]:



epochscount = 300
num_batch_size = 32

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochscount, validation_data=(X_test, y_test), verbose=1)
validation_test_set_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(validation_test_set_accuracy[1])


# In[35]:


validation_test_set_accuracy = model.evaluate(X_test,y_test,verbose=0)
print(validation_test_set_accuracy[1])


# In[42]:


X_test[1]


# In[43]:


predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)


# In[44]:


classes_x


# In[45]:


#predict


# In[46]:


filename='C:/Users/user/Desktop/Ödev/yapayZeka/PoliceSiren.wav'
sound_signal, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[47]:


print(mfccs_scaled_features)


# In[48]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[49]:


mfccs_scaled_features.shape


# In[50]:


print(mfccs_scaled_features)


# In[51]:


print(mfccs_scaled_features.shape)


# In[52]:


result_array = model.predict(mfccs_scaled_features)


# In[54]:


result_array


# In[55]:


result_classes = ["air_conditioner","car_horn","children_playing","dog_bark","drilling", "engine_idling", 
                  "gun_shot", "jackhammer", "siren", "street_music"]
result = np.argmax(result_array[0])
print(result_classes[result]) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




