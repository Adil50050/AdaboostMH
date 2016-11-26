#!/usr/bin/python3
"""
Implement of Adaboost.MH
"""
import numpy as np
from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score
import math

class AdaboostMH:
    def __init__(self, filename, T):
        # 弱分类器最大数目
        self.T = T

        # 原始数据，前70%作为训练数据，后30%作为测试数据
        self.rawdata = []
        self.rawlabel = []
        self.testdata = []
        self.testlabel = []


        # 扩展后的数据集和对应的类，转化为对应的二分类问题
        self.data = [] # processed data
        self.target = []

        # 从数据文件中读取数据，并扩展数据
        self.loaddata(filename)

        # 弱分类器
        self.classifier = []

        # 样本权重
        self.D = np.array([1. / len(self.target) for i in range(len(self.target))])

        # 分类器权重
        self.alpha = []

    def loaddata(self, filename):
        datafile = open(filename, 'r')
        for line in datafile.readlines():
            if line[-1] == '\n':
                line = line[:-1]
            splited_line = line.split(',')
            for i in range(len(splited_line)):
                if splited_line == 'None':
                    splited_line[i] = -1
            self.rawdata.append(list(map(int,splited_line[1:])))
            self.rawlabel.append(ord(splited_line[0]))

        self.labels = list(set(np.array(self.rawlabel)))
        idx = int(len(self.rawlabel) * 0.7)
        self.testdata = self.rawdata[idx:]
        self.testlabel = self.rawlabel[idx:]
        self.rawdata = self.rawdata[:idx]
        self.rawlabel = self.rawlabel[:idx]

        for i in range(len(self.rawlabel)):
            for lable in self.labels:
                self.data.append(list(self.rawdata[i]))
                self.data[-1].append(lable)
                if self.rawlabel[i] == lable:
                    self.target.append(1)
                else:
                    self.target.append(-1)

    def train(self):
        for i in range(self.T):
            print("Training for the %dth weak classifier..." % (i+1))
            # 训练弱分类器
            cur_clf = LinearSVC()
            cur_clf.fit(self.data, self.target, sample_weight=self.D)

            # 计算错误率
            results, error_rate = self.get_error_rate(cur_clf)
            print("Error rate is %f" % (error_rate))

            # 计算弱分类器权重
            alpha = math.log(1/error_rate - 1) / 2
            print("Alpha is %f" % (alpha))

            # 弱分类器错误率大于0.5，停止训练
            if error_rate >= 0.5:
                print("Error rate is larger than 0.5, stop training...")
                break

            self.classifier.append(cur_clf)
            self.alpha.append(alpha)

            # 更新样本权重
            self.update_weight(alpha, results)

    def get_error_rate(self, clf):
        results = clf.predict(self.data)
        error_rate = 1 - accuracy_score(self.target, results)
        return [results, error_rate]

    def update_weight(self, alpha, results):
        for i in range(len(self.data)):
            if results[i] == self.target[i]:
                self.D[i] = self.D[i] * math.exp(-alpha)
            else:
                self.D[i] = self.D[i] * math.exp(alpha)
        z = sum(self.D)
        for i in range(len(self.D)):
            self.D[i] /= z

    def test(self):
        print("Test sample num: %d" % (len(self.testlabel)))

        # 拓展测试数据集
        data_ext = []
        label_ext = []
        for i in range(len(self.testdata)):
            #print(self.testdata[i])
            for j in range(len(self.labels)):
                tmpdata = list(self.testdata[i])
                tmpdata.append(self.labels[j])
                #print(tmpdata)
                data_ext.append(tmpdata)
                if self.labels[j] == self.testlabel[i]:
                    label_ext.append(1)
                else:
                    label_ext.append(-1)
        predict = []
        for clf in self.classifier:
            print(clf.classes_)
            curpredict = clf.predict_proba(data_ext)
            predict.append(curpredict)
        count = 0
        fuck = []
        for i in range(len(label_ext)):
            # print(label_ext[i], predict[0][i])
            if predict[0][i][1] < predict[0][i][0]:
                fuck.append(-1)
            else:
                fuck.append(1)
        print('Error rate:', accuracy_score(label_ext, fuck))
        # print(predict[0][0])
        allret = []
        for i in range(len(data_ext)):
            ret = np.array([0., 0.])
            for j in range(len(self.classifier)):
                # print(self.alpha[j], predict[j][i])
                ret += (self.alpha[j] * predict[j][i])
            # print(ret)
            allret.append(ret)

        result = []
        for i in range(len(self.testlabel)):
            curmax = 0
            curlabel = 0
            # print("for test sample %d" % (i))
            # print("Raw data is:", self.testdata[i])
            # print("Should be:", self.rawlabel[i])
            for j in range(len(self.labels)):
                # print("Extened data is:", data_ext[i * len(self.labels) + j])
                # print("CLF 0:", predict[0][i * len(self.labels) + j])
                # print("CLF 1:", predict[1][i * len(self.labels) + j])
                # print(allret[i * len(self.labels) + j])
                if allret[i * len(self.labels) + j][1] > curmax:
                    curmax = allret[i * len(self.labels) + j][1]
                    curlabel = self.labels[j]
            result.append(curlabel)
            # print("Result is:", curlabel)
        error_num = 0
        for i in range(len(result)):
            if self.testlabel[i] != result[i]:
                error_num += 1
        print("Error rate is %f" % (float(error_num)/len(result)))

if __name__ == '__main__':
    adaboostMH = AdaboostMH('letter-recognition.data', 1)
    adaboostMH.train()
    adaboostMH.test()
