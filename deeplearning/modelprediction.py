import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
from keras.engine.saving import load_model
sns.set(color_codes=True)


#model = load_model('trainedmodel_coughs.h5')
model = load_model('trainedmodel_fus.h5')


# Data from csv to pandas DataFrame
mg_data = pd.DataFrame()
# threshold - 0.16
mg_data = pd.read_pickle('/home/pinkie/pydeeplearn_anomaly/dataset_hol.pkl')

# threshold - 0.16
#mg_data = pd.read_pickle('/home/pinkie/pydeeplearn_anomaly/dataset_fus.pkl')

# Defining train/test data
train = mg_data[0:]
test = mg_data[0:]

# Normalize
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['ztráta'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['threshold'] = 0.1
scored['anomálie'] = scored['ztráta'] > scored['threshold']
scored.loc[scored['ztráta'] > scored['threshold'], 'isAnomaly'] = scored['threshold'] + 0.6
scored.loc[scored['ztráta'] < scored['threshold'], 'isAnomaly'] = scored['threshold'] + 0.5
print(scored.head)
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red', 'black'], marker='')
plt.show()
