import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Visualization.Feature_importance import FeatImp


data = pd.read_csv('data/avocado.csv')
print(data.dtypes)
data_1 = data.drop(['Date','type','region'],axis =1)
fi = FeatImp(data_1)
fi.tree_selection(target_name='AveragePrice')
fi.corr_matrix(target_name='AveragePrice',method_version=2)