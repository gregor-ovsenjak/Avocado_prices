import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Visualization.Feature_importance import FeatImp
import re



data = pd.read_csv('data/avocado.csv')

data_1 = data.drop(['Date','type','region'],axis =1)
