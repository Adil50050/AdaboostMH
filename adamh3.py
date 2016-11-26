#!/usr/bin/python3
"""
Implement of Adaboost.MH
"""
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import math

x_train = []
y_train = []

x_test = []
y_test = []

def loaddata(filename):
    datafile = open(filename, 'r')
    x = []
    y = []
    for line in datafile.readlines():
        if line[-1] == '\n':
            line = line[:-1]
        splited = line.split(',')
        tmpx = list(map(int, splited[:-1]))
        tmpy = int(splited[-1])
        x.append(tmpx)
        y.append(tmpy)
    return [np.array(x), np.array(y)]


x_train, y_train = loaddata('train_ext.data')
x_test, y_test = loaddata('test_ext.data')
print("------ Traning ------")
print("Sample num: %d" % (len(x_train)/26))
clf = AdaBoostClassifier()
clf.fit(x_train, y_train)

print('------ Testing ------')
print('Sample num: %d' % (len(x_test)/26))

rawtx, rawty = loaddata('test.data')

predict = clf.predict_proba(x_test)
fuck = clf.predict(x_test)
result = []
for i in range(len(rawtx)):
    curmax = predict[i*26][1]
    curlabel = 0
    print("Raw data:", rawtx[i], rawty[i])
    for j in range(26):
        print(j, fuck[i*26+j], predict[i*26+j])
        if predict[i*26 + j][1] >= curmax:
            curmax = predict[i*26 + j][1]
            curlabel = j
    result.append(j)

count = 0
for i in range(len(result)):
    if rawty[i] != result[i]:
        count+=1
print("Final error rate is:", float(count)/float(len(result)))
