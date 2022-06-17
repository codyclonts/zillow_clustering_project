### this file holds functions that are used in zillow clustering project###

import pandas as pd 
import numpy as np 
import wrangle_zillow as wrangle

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from scipy import stats 
from itertools import combinations
import wrangle_zillow as wrangle
from sklearn.model_selection import train_test_split



df = wrangle.wrangle_zillow()


train, validate, test = wrangle.split_zillow_data(df)

####################### Functions info #######################

columns = [
 'bathroomcnt',
 'bedroomcnt',
 'calculatedfinishedsquarefeet',
 'fips',
 'latitude',
 'longitude',
 'lotsizesquarefeet',
 'regionidcity',
 'regionidzip',
 'yearbuilt',
 'taxvaluedollarcnt',
 'taxamount',
 'id.1',
 'logerror',
 'month'
]

############################# visual functions#############################
target = 'logerror'
plot_hue = 'fips'
# this function cycles through the columns and makes an lm plot for every pair of variables
def plot_variable_pairs():
    for col in columns:
        sns.lmplot(x = col, y = target , hue = plot_hue, data = train, x_bins = 20)




### this function plots the distribution of each variable in a histogram
def variable_dist():
    for col in train.columns:
        plt.figure(figsize=(4,2))
        plt.hist(train[col])
        plt.title(col)
        plt.show()

        