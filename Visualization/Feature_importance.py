import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from itertools import cycle, islice
import re



# function for seperating target values and feature values
def seperate_target_feature(data,target_name):
    y = data[target_name]
    X = data.drop(target_name,axis = 1)
    return X,y



class FeatImp():

    def __init__(self,data,classifier = ExtraTreesRegressor()):
        self.data = data
        self.model = classifier # tree based classifiers are used for visualizing Feature importance



    def tree_selection(self,target_name):
        # plots a horizontal bar graph of most important features, selected by Extra
        # TreesClassifier. You can choose your own classifier as long as it is tree based
        X,y = seperate_target_feature(self.data,target_name)
        self.model.fit(X,y)
        # colors of the bars 
        my_colors = list(islice(cycle(['blue', 'red', 'green', 'gold', 'm','olive','purple','cyan','maroon']), None, len(X)))
        
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(self.model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh',stacked=True,color = my_colors)
        plt.show()
    
    def corr_matrix(self,target_name,method_version = 0):
        methods = ["pearson","kendall","spearman"]
        
        X,y = seperate_target_feature(self.data,target_name)
        #get correlations of each features in dataset
        correlation_matrix = self.data.corr(method = methods[method_version])
        top_corr_features = correlation_matrix[target_name].nlargest(10).index.values
        plt.figure(figsize=(10,10))
        #plot heat map
        g=sns.heatmap(self.data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        plt.show()
    
    def statistical_test(self,target_name):
        #only for non-negative features
        X,y = seperate_target_feature(self.data,target_name)
        #apply SelectKBest class to extract top 10 best features
        X_new = SelectPercentile(chi2, percentile=10).fit(X, y)
        dfscores = pd.DataFrame(X_new.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        print(featureScores)  #print 10 best features