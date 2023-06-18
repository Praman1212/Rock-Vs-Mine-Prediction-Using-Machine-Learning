# Rock-Vs-Mine-Prediction-Using-Machine-Learning
Using LogisticRegression model this project helps to predict whether the given data is rock or mine. 
Here is the complete code of this project: 
I am using dataset available on kaggle name as rock vs mine

Code: 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# numpy is for arrays where pandas is for data processing or cleaning.

# Data Collection and Data processing 
sonar_data = pd.read_csv('sonar_data.csv',header=None)

sonar_data.head()
sonar_data.shape

sonar_data.describe()
# Describe the statical measures of the data

sonar_data[60].value_counts()
# Shows the 60th index of sonar_data count
# M-> mines
# R->Rock
sonar_data.groupby(60).mean()

# Seperating data and Labels
X = sonar_data.drop(columns = 60, axis = 1)
Y = sonar_data[60]

# Training and Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.1, stratify=Y, random_state =1)

# Model training using logistic Regression Model
model = LogisticRegression()

# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Model Evaluation: calculating accuracy of training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ', training_data_accuracy)

# Model Evaluation: calculating accuracy of tested data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ', test_data_accuracy)

# Making predictive system
input_data = (0.0270,0.0163,0.0341,0.0247,0.0822,0.1256,0.1323,0.1584,0.2017,0.2122,0.2210,0.2399,0.2964,0.4061,0.5095,0.5512,0.6613,0.6804,0.6520,0.6788,0.7811,0.8369,0.8969,0.9856,1.0000,0.9395,0.8917,0.8105,0.6828,0.5572,0.4301,0.3339,0.2035,0.0798,0.0809,0.1525,0.2626,0.2456,0.1980,0.2412,0.2409,0.1901,0.2077,0.1767,0.1119,0.0779,0.1344,0.0960,0.0598,0.0330,0.0197,0.0189,0.0204,0.0085,0.0043,0.0092,0.0138,0.0094,0.0105,0.0093)

# chaning the input_data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for tone instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if(prediction[0] == 'R'):
    print('The object is Rock')
else:
    print('The boject is Mine')
