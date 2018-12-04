import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import  normalize, StandardScaler
import matplotlib.pyplot as plt


def rmsleCalc(targets, predictions):
    sum=0.0
    for x in range(len(predictions)):
        if predictions[x]<0 or targets[x]<0: #check for negative values
            continue
        p = np.log(predictions[x]+1)
        r = np.log(targets[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predictions))**0.5



dataframe = pd.read_csv('train.csv')
dataframe = pd.get_dummies(dataframe)

last_col = dataframe.pop('SalePrice')   #just rearranging columns for my convinience
dataframe['SalePrice'] = last_col       #

dataset = dataframe.values

features = np.nan_to_num(dataset[:, 0:len(dataset[0])-2])
target = dataset[:, len(dataset[0])-1]

# INSERT NORMALIZATION OR SCALING HERE IF SOMETHING IS WRONG
features = StandardScaler().fit_transform(features)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

model = linear_model.Lasso(alpha=10,normalize=True)              #Normalization?
model.fit(features_train, target_train)

scores = cross_val_score(model, features_train, target_train, cv=5)
target_pred = model.predict(features_test)

print("Cross validation score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print('RMSLE: %.2f' % rmsleCalc(target_test, target_pred))
print('Variance score: %.2f' % metrics.r2_score(target_test, target_pred))