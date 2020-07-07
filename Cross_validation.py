import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold




class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]



def build_models():

    classifiers = []
    classifiers.append(SVR())
    classifiers.append(RandomForestRegressor())
    classifiers.append(ExtraTreesRegressor())
    classifiers.append(AdaBoostRegressor(base_estimator=DecisionTreeRegressor()))
    classifiers.append(LinearRegression())
    classifiers.append(SGDRegressor())

    return classifiers

def cross_validation(X_train,y_train):
    model_names = ['SVR','Random Forest Regressor','Extra Trees Regressor','AdaBoost','LinearRegresion','SGD']
    models = build_models()
    btscv = BlockingTimeSeriesSplit(n_splits=5)
    cv_results = []
    for model in models:
        cv_results.append(cross_val_score(model, X_train, y_train, cv=btscv, scoring='r2'))
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(cv_result.mean())
        cv_std.append(cv_result.std())
    df_cv_results = pd.DataFrame(data=[cv_means,cv_std],index=["means","std"],columns = model_names)
    return df_cv_results