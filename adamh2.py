#!/usr/bin/python3
"""
Implement of Adaboost.MH
"""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import math
T = 1
classifier = []
alpha = []

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
D = np.array([1./len(x_train) for i in range(len(x_train))])

#clf = AdaBoostClassifier()
#clf.fit(x_train, y_train)
#print(clf.classes_)
#print("Score for Adaboost.MH is:", clf.score(x_test, y_test))

def updateWeight(clf, ialpha):
    result = clf.predict(x_train)
    for i in range(len(x_train)):
        if y_train[i] == result[i]:
            D[i] = D[i] * math.exp(-ialpha)
        else:
            D[i] = D[i] * math.exp(ialpha)
    z = sum(D)
    for i in range(len(D)):
        D[i] /= z


print('------ Training ------')
print('Sample num: %d' % (len(x_train)/26))
for i in range(T):
    print("Training for the %dth weak classifier..." % (i))
    clf = DummyClassifier()
    clf.fit(x_train, y_train, D)
    error_rate = 1. - clf.score(x_train, y_train, sample_weight = D)
    print('Error rate is:', error_rate)
    if (error_rate >= 0.5):
        print('Error rate >= 0.5, stop training...')

    ialpha = math.log(1.0 / error_rate - 1) / 2
    print('Alpha is:', ialpha)
    classifier.append(clf)
    alpha.append(ialpha)
    updateWeight(clf, ialpha)
    print('Sample weight updated')

print('------ Testing ------')
print('Sample num: %d' % (len(x_test)/26))
predict = np.zeros((len(x_test), 2))
rawtx, rawty = loaddata('test.data')

for i in range(len(classifier)):
    print("CLF", i, "score:", accuracy_score(y_test, classifier[i].predict(x_test)))
    curpre = classifier[i].predict_proba(x_test)
    curpre = curpre.astype('float64')
    curpre *= alpha[i]
    predict += curpre
fuck1 = classifier[0].predict(x_test)
#fuck2 = classifier[1].predict(x_test)
result = []
for i in range(len(rawtx)):
    curmax = predict[i*26][1]
    curlabel = 0
    print("Raw data:", rawtx[i], rawty[i])
    for j in range(26):
        print(j, fuck1[i*26+j], predict[i*26+j])
        if predict[i*26 + j][1] >= curmax:
            curmax = predict[i*26 + j][1]
            curlabel = j
    result.append(j)

count = 0
for i in range(len(result)):
    if rawty[i] != result[i]:
        count+=1
print("Final error rate is:", float(count)/float(len(result)))
