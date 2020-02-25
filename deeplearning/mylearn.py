import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from array import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

import seaborn as sns
sns.set(color_codes=True)

def getColumns(count):
    myarray = []
    for x in range(count):
        myarray.append(str(x))
    return myarray

def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

# Data from csv to pandas DataFrame
mg_data = pd.DataFrame()

dataset = pd.read_csv(os.path.join('/home/pinkie/pydeeplearn', 'pointshol.csv'), sep="\t", index_col=0)
#dataset = pd.read_csv(os.path.join('/home/pinkie/pydeeplearn', 'pointsfus.csv'), sep="\t", index_col=0)
dataset_fin = np.array(dataset)

dataset_index = 0
for x in dataset_fin:
    dataset_fin = pd.DataFrame(x.reshape(1,210))
    dataset_fin.index = [dataset_index]
    dataset_index += 1
    mg_data = mg_data.append(dataset_fin)

mg_data.columns = getColumns(210)

# print(mg_data.head())

mg_data.to_pickle('/home/pinkie/pydeeplearn/dataset_hol.pkl')

# Defining train/test data
train = mg_data[0:275]
test = mg_data[276:]


# fig, ax = plt.subplots(figsize=(11, 6), dpi=80)
# ax.plot(train["198"])
# ax.plot(train["186"])
# plt.legend(loc='lower left')
# ax.set_title('Bearing Sensor Training Data', fontsize=16)
# plt.show()


scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()



# fit the model to the dat
nb_epochs = 100
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.1).history



# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.5])

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.15
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])
plt.show()



# save all model information, including weights, in h5 format
model.save("trainedmodel_fus.h5")
print("Model saved")



# plt.plot(mp62)
# plt.plot(mp66)
# plt.ylabel('some numbers')
# plt.show()


