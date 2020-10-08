#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
from tqdm import tqdm
from typing import Counter
import math
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, precision_score
import sys
import time
import warnings
from sklearn.ensemble import GradientBoostingClassifier
import threading


def load_data():
    data = loadmat("../data/mnist_all.mat")

    # print(data.keys())

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for i in range(10):
        temp_df = pd.DataFrame(data["train" + str(i)])
        temp_df['label'] = i
        train_data = train_data.append(temp_df)
        temp_df = pd.DataFrame(data["test" + str(i)])
        temp_df['label'] = i
        test_data = test_data.append(temp_df)

    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

    train_labels = np.array(train_data['label'])
    test_labels = np.array(test_data['label'])

    train_data = train_data.drop('label', axis=1)
    test_data = test_data.drop('label', axis=1)
    
    train_data = np.array(train_data) / 255
    test_data = np.array(test_data) / 255
    
    pca = PCA(0.95)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)

    return train_data, test_data, train_labels, test_labels


class RegressionTree:
    def __init__(self, min_samples_leaf, K, max_depth=3):
        self.min_samples_leaf = min_samples_leaf
        self.K = K
        self.max_depth = max_depth
    
    def fit(self, x, y):
        depth = 0
        self.tree = self.build_tree(x, y, depth)
        
    
    def build_tree(self, x, y, depth):
        
        best_feature_index, threshold, c1, c2 = self.choose_best_feature_to_split(x, y)
        
        tree = {}
        x1, y1, x2, y2 = self.split_data(x, y, best_feature_index, threshold)
        
        
        # 构造树的终止条件
        if len(x1) < self.min_samples_leaf or depth >= self.max_depth:
            # 这里需要符合gbdt的公式
            sum1 = sum(y1)
            sum2 = sum([abs(_y)*(1-abs(_y)) for _y in y1])
            
            if sum1 == 0 or sum2 == 0:
                v = 0
            else:
                v = ((self.K-1)/self.K)*(sum1/sum2)
                
            tree[(best_feature_index, threshold, '<=')] = v
        else:
            tree[(best_feature_index, threshold, '<=')] = self.build_tree(x1, y1, depth + 1)
            
        if len(x2) < self.min_samples_leaf or depth >= self.max_depth:
            # 这里需要符合gbdt的公式
            sum1 = sum(y2)
            sum2 = sum([abs(_y)*(1-abs(_y)) for _y in y2])
            
            if sum1 == 0 or sum2 == 0:
                v = 0
            else:
                v = ((self.K-1)/self.K)*(sum1/sum2)
            
            tree[(best_feature_index, threshold, '>')] = v
        else:
            tree[(best_feature_index, threshold, '>')] = self.build_tree(x2, y2, depth + 1)
            
        return tree
    
            
    def split_data(self, x, y, best_feature_index, threshold):
        
        x1, x2, y1, y2 = [], [], [], []
        
        for i in range(len(x)):
            if x[i][best_feature_index] <= threshold:
                x1.append(x[i])
                y1.append(y[i])
            else:
                x2.append(x[i])
                y2.append(y[i])
        
        return np.array(x1), np.array(y1), np.array(x2), np.array(y2)
    
    
    def calculate_mse(self, feature_index, x, y):
        values = []
        for i in range(len(x)):
            values.append(x[i][feature_index])
        
        values = list(set(values))
        
        n1, n2 = 0, 0
        y1, y2 = 0, 0
        
        best_mse = sys.maxsize
        best_threshold = None
        best_c1, best_c2 = None, None
        
        for value in values:
            for i in range(len(x)):
                if x[i][feature_index] <= value:
                    n1 += 1
                    y1 += y[i]
                elif x[i][feature_index] > value:
                    n2 += 1
                    y2 += y[i]
            
            if n1 != 0:
                c1 = y1/n1
            else:
                c1 = 0
            
            if n2 != 0:
                c2 = y2/n2
            else:
                c2 = 0
            
            mse = 0
            for i in range(len(x)):
                if x[i][feature_index] <= value:
                    mse += (c1 - y[i])*(c1 - y[i])
                elif x[i][feature_index] > value:
                    mse += (c2 - y[i])*(c2 - y[i])
                    
            if mse < best_mse:
                best_mse = mse
                best_threshold = value
                best_c1 = c1
                best_c2 = c2
        
        # 不会发生
        if best_threshold is None:
            pass
        
        return best_mse, best_threshold, best_c1, best_c2
    
    
    def choose_best_feature_to_split(self, x, y):
        n_features = x.shape[1]
        
        best_feature_index = -1
        best_mse = sys.maxsize
        best_feature_threshold = None
        best_c1 = None
        best_c2 = None
        
        for feature_index in range(n_features):
            mse, threshold, c1, c2 = self.calculate_mse(feature_index, x, y)
            
            if mse < best_mse:
                best_feature_index = feature_index
                best_mse = mse
                best_feature_threshold = threshold
                best_c1 = c1
                best_c2 = c2
        # 不会发生
        if best_feature_index == -1:
            pass
        
        return best_feature_index, best_feature_threshold, best_c1, best_c2
    
    
    def predict_value(self, x):
        tree = self.tree
        
        while type(tree).__name__ == 'dict':
            
            for key in tree.keys():
                if key[2] == '<=':
                    key1 = key
                elif key[2] == '>':
                    key2 = key
                    
            
            feature_index = key1[0]
            threshold = key1[1]
            
            if x[feature_index] <= threshold:
                tree = tree[key1]
            elif x[feature_index] > threshold:
                tree = tree[key2]

        
        if type(tree).__name__ != 'dict':
            return tree
        else:
            pass
    
    
    def predict(self, X):
         return np.array([self.predict_value(x) for x in X])



class GBDT:
    def __init__(self, n_estimators, learning_rate, max_depth=3, min_samples_leaf=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
        self.trees = {}
    
    def fit(self, x, y):
        
        K = len(list(set(y)))
        
        self.K = K
        
        y = [self.to_one_hot(K, _y) for _y in y]
        
        F = {}
        p = {}
        residual = {}
        for i in range(len(x)):
            F[i] = {}
            p[i] = {}
            residual[i] = {}
            for k in range(K):
                F[i][k] = 0.0
                
        for m in tqdm(range(self.n_estimators)):
            
            self.trees[m] = {}
            
            for i in range(len(x)):
                denominator = sum([np.exp(F[i][_k]) for _k in range(K)])
                for k in range(K):
                    p[i][k] = np.exp(F[i][k])/denominator
                    residual[i][k] = y[i][k] - p[i][k]
                    
            for k in range(K):
                residuals = []
                for i in range(len(x)):
                    residuals.append(residual[i][k])
                    
                tree = RegressionTree(min_samples_leaf=self.min_samples_leaf, K=K)
                
                tree.fit(x, residuals)
                
                self.trees[m][k] = tree
                    
                # update F
                for i in range(len(x)):
                    F[i][k] += self.learning_rate * self.trees[m][k].predict_value(x[i])
                    
    
    def predict_value(self, x):
        p = [0]*self.K
        for m in self.trees:
            for k in range(self.K):
                p[k] += self.learning_rate * self.trees[m][k].predict_value(x)
        
        return np.argmax(p)
    
    
    def predict(self, X):
        return np.array([self.predict_value(x) for x in X])
    
    
    def to_one_hot(self, n_class, x):
        t = np.zeros(shape=(n_class, 1))
        t[x] = 1
        return t


def main():
    warnings.filterwarnings("ignore", category=Warning)

    X_train, X_test, y_train, y_test = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0)


    # 注意，经过PCA降维后，认为所有的特征都是连续值
    X = np.concatenate((X_train, X_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)

    print(X.shape, y.shape)


    sklearn_gbdt = GradientBoostingClassifier(n_estimators=2, max_depth=3)
    sklearn_gbdt.fit(X[:100], y[:100])
    predict_sklearn = sklearn_gbdt.predict(X_test)
    print('sklearn gbdt test acc: {}'.format((sum(predict_sklearn == np.array(y_test)))/len(X_test)))

    custom_gbdt = GBDT(n_estimators=2, learning_rate=0.001)
    print('start training custom gbdt...')
    start = time.time()
    custom_gbdt.fit(X[:100], y[:100])
    end = time.time()
    print('finish training... time cost: {}s'.format(end-start))
    predict_custom = custom_gbdt.predict(X_test)
    print('custom gbdt test acc: {}'.format((sum(predict_custom == np.array(y_test)))/len(X_test)))


if __name__ == '__main__':
    main()




