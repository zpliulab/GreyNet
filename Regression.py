from  sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
from sklearn.metrics import roc_curve,roc_auc_score,auc,precision_recall_curve
import numpy as np
import pandas as pd

class Regression:
    def __init__(self,data,type='RF'):
        self.data = data.values
        self.m, self.n = data.shape
        self.type = type
        self.columns = data.columns.values


    def bagging(self):
        model = RandomForestRegressor(n_estimators=1000, max_features='sqrt')

        VIM = np.zeros([self.n, self.n])
        for index in range(self.n):
            x_train, y_train = self.data_process(index=index)
            model.fit(x_train, y_train)
            imp = model.feature_importances_
            imp = np.insert(imp, index, 0)
            VIM[:, index] = imp
        return VIM

    def boosting(self):
        model = XGBRegressor(lr=0.0001, max_depth=3, ntrees=1000,
                              subsample=0.8, bylevel=0.6, bytree=0.6)

        VIM = np.zeros([self.n, self.n])
        for index in range(self.n):
            x_train, y_train = self.data_process(index=index)
            model.fit(x_train, y_train)
            imp = model.feature_importances_
            imp = np.insert(imp, index, 0)
            VIM[:, index] = imp
        return VIM

    def lasso(self):
        model = Lasso(alpha=0.01,max_iter=1000, normalize=True)

        VIM = np.zeros([self.n, self.n])
        for index in range(self.n):
            x_train, y_train = self.data_process(index=index)
            model.fit(x_train, y_train)
            imp = model.coef_
            imp = abs(imp)
            imp = np.insert(imp, index, 0)
            VIM[:, index] = imp
        return VIM

    def ridge(self):
        model = Ridge(alpha=1, max_iter=1000, normalize=True)

        VIM = np.zeros([self.n, self.n])
        for index in range(self.n):
            x_train, y_train = self.data_process(index=index)
            model.fit(x_train, y_train)
            imp = model.coef_
            imp = abs(imp)
            imp = np.insert(imp, index, 0)
            VIM[:, index] = imp
        return VIM

    def data_process(self,index, lag=1):

        if lag >= self.data.shape[0]:
            raise ValueError(r'the lag is out of range')

        data_dul = self.data.copy()

        m, n = data_dul.shape

        target = data_dul[:,index]

        regulators = np.delete(data_dul, index, axis=1)

        target = target[lag:m]
        regulators = regulators[:m - lag, :]

        target = zscore(target)
        regulators = zscore(regulators)

        return regulators, target

    def regu_tar(self,columns):
        index = []
        for regu in columns:
            for tar in columns:
                ind = '(' + regu + ',' + tar + ')'
                index.append(ind)
        blank = pd.Series(np.zeros(len(index)), index=index)
        return blank

    def gene_pairwise(self,VIM,columns):
        VIM_vector = VIM.flatten()
        pred = self.regu_tar(columns)
        pred[pred.index.values] = VIM_vector

        return pred

    def fit(self):
        if self.type == 'RF':
            vim = self.bagging()
        elif self.type == 'Xgboost':
            vim = self.boosting()
        elif self.type == 'Lasso':
            vim = self.lasso()
        elif self.type == 'Ridge':
            vim = self.ridge()
        else:
            raise TypeError('{} not include in [RF, Xgboost,Lasso,Ridge]'.format(self.type))

        y_prob = self.gene_pairwise(vim, self.columns)

        return y_prob





