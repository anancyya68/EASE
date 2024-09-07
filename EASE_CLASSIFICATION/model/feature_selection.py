# @File : feature_selection.py
# @Time : 2024/5/20 17:24
# @Author :

import pandas as pd
import random
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV,KFold
import math


class FeatureSelection:
    def __init__(self, estimator, predictor):
        self.estimator = estimator
        self.predictor = predictor

    def parameter_search(self,X_val, y_val, predictor, candidate_params, n_splits=3):
        """
        :param X_val: features
        :param y_val: label
        :param train_algorithm: algorithm
        :param train_parameters: parameters
        :return select_parameters:parameters
        """
        best = GridSearchCV(predictor, candidate_params,cv=KFold(n_splits=n_splits, shuffle=True, random_state=22))
        best.fit(X_val, y_val)
        return best.best_estimator_

    def rfe_method_class(self, X_train, y_train, X_test, y_test, X_val, y_val, candidate_params, n_features,step=1, verbose=0):
        """
        RFE for classification
        """
        # feature transform
        rfe = RFE(estimator=self.estimator, n_features_to_select=n_features, step=step, verbose=verbose)
        rfe.fit(X_train, y_train)
        X_train_selected = rfe.transform(X_train)
        X_test_selected = rfe.transform(X_test)
        X_val_selected = rfe.transform(X_val)

        # parameter search and train
        best_predictor = self.parameter_search(X_val_selected, y_val,self.predictor, candidate_params)
        best_predictor.fit(X_train_selected, y_train)

        # prediction
        y_pred = best_predictor.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        return acc, pre, f1, recall

    def rfe_method_class_time(self, X_train, y_train, X_test, y_test, X_val, y_val, candidate_params, n_features,step=1, verbose=0):
        """
        RFE for classification
        """
        # feature transform

        best_estimator = self.parameter_search(X_val, y_val, self.estimator, candidate_params)
        best_estimator.fit(X_train, y_train)


        rfe = RFE(estimator=self.estimator, n_features_to_select=n_features, step=step, verbose=verbose)
        rfe.fit(X_train, y_train)
        X_train_selected = rfe.transform(X_train)
        X_test_selected = rfe.transform(X_test)
        X_val_selected = rfe.transform(X_val)

        # parameter search and train
        best_predictor = self.parameter_search(X_val_selected, y_val,self.predictor, candidate_params)
        best_predictor.fit(X_train_selected, y_train)

        # prediction
        y_pred = best_predictor.predict(X_test_selected)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        return acc, pre, f1, recall


