import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder






class Log_Transform(BaseEstimator, TransformerMixin):
    #log transformation of numerical features 
    def __init__(self,columns):
        self.columns = columns

    def fit(self,data,y=None):
        return self
    
    def transform(self,data,y=None):
        data += 1
        data = np.log10(data)
        return data


class DataFrameSelector(BaseEstimator,TransformerMixin):
    # returns only values from a DataFrame
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self,X,y = None):
        return self

    def transform(self,X,y = None):
        return X[self.attribute_names].values



class Transformations(BaseEstimator,TransformerMixin):
    '''class for numerical and categorical transformations, takes in numerical column names,
       and categorical column names. Applies log-transformation
       to numerical attributes and OneHotEncoding on categorical attributes.
    '''
    def __init__(self,number_col,category_col):

        self.number_col = number_col
        self.category_col = category_col
        self.encoder = OneHotEncoder(sparse=False)
        self.number_pipeline = Pipeline([('dataFS_cat',DataFrameSelector(self.number_col)),
                                        ('Log',Log_Transform(self.number_col)),
                                        ])
        self.cat_pipeline = Pipeline([ ('dataFS_cat',DataFrameSelector(self.category_col)),
                                       ('label_binarizer', self.encoder),    
                                    ])
        self.full_pipeline = FeatureUnion([
                                            ("num_pipe",self.number_pipeline),
                                            ("cat_pipe",self.cat_pipeline),
                                        ])

    def fit_and_transform(self,X,y = None,only_transform = False):
        if not only_transform:
            return self.full_pipeline.fit_transform(X,y = None)
        else:
            return self.full_pipeline.transform(X)



def transform_date_to_month(data):
    # Changes dtype of 'year' feature from int to object
    data['year'] = data['year'].apply(str)
    # gets only month values from 'Date' feature
    date_regex = r'-([0-9]{2,2})-'
    months = re.findall(date_regex,data['Date'].values.sum())
    data['month'] = months
    # drops columns that are no longer of use 
    data = data.drop(['Date'],axis = 1)
    return data



def back_to_Data_Frame(data):

    new_col_names = []
    # appending all unique values of each feature
    for column_name in data.columns:
        # if dtype == int,float..., then append feature name
        if data[column_name].dtype == np.number:
            new_col_names.append(column_name)
        # if dtype == object, then append each unique value of feature
        elif data[column_name].dtype == np.object:
            new_col_names.extend(sorted(data[column_name].unique()))
    return new_col_names



def extract_cat_num_names(data):
    #get categorical and numerical names
    categorical_col = data.select_dtypes([np.object]).columns
    number_col = data.select_dtypes([np.number]).columns
    return categorical_col,number_col



def apply_transformations(data):
    '''Apply transformations in order:
     --> transform Date feature --> month feature
     --> log-transformation to numerical features, OneHotEncoder to cat features
     --> return np.array() back to DataFrame with new column names
    '''
    data = transform_date_to_month(data)
    cat_names,num_names = extract_cat_num_names(data)
    transformer = Transformations(num_names,cat_names)
    data_1 = transformer.fit_and_transform(data)
    return pd.DataFrame(data_1, columns= back_to_Data_Frame(data))

