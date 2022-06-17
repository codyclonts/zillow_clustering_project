# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats

# Visualizing

import matplotlib.pyplot as plt
import seaborn as sns

from env import get_db_url
import os
from sklearn.model_selection import train_test_split

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format
from sklearn.preprocessing import MinMaxScaler



######################################

def get_zillow_data():
    filename = 'zillow_data.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    else:
        df = pd.read_sql(
        '''
        SELECT *
        FROM properties_2017
        JOIN propertylandusetype 
        USING (propertylandusetypeid)
        JOIN predictions_2017
        USING (parcelid)
        WHERE propertylandusedesc = 'Single Family Residential';
        '''
        ,
        get_db_url('zillow')
        )
        
        df.to_csv(filename)
        
        return df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column * len(df.index), 0))
    df.dropna(axis=1, thresh = threshold, inplace = True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df
    



def split_zillow_data(df):

    train_validate, test = train_test_split(df, test_size=.2, 
        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
        random_state=123)
    return train, validate, test


def data_scaled(train, validate, test, columns_to_scale):
    '''
    This function takes in train, validate, test subsets of the cleaned zillow dataset and using the train subset creates a min_max 
    scaler. It thens scales the subsets and returns the train, validate, test subsets as scaled versions of the initial data.
    Arguments:  train, validate, test - split subsets from of the cleaned zillow dataframe
                columns_to_scale - a list of column names to scale
    Return: scaled_train, scaled_validate, scaled_test - dataframe with scaled versions of the initial unscaled dataframes 
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = MinMaxScaler()
    
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])

    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    return train_scaled, validate_scaled, test_scaled

def make_int(df):
    df = df.astype({'bedroomcnt':'int', 'calculatedfinishedsquarefeet' : 'int' , 'bathroomcnt' : 'int', 'propertylandusetypeid' : 'int' , 'fips' : 'int' , 'latitude' : 'int' , 'longitude' : 'int' , 'lotsizesquarefeet' : 'int' , 'regionidcity' : 'int' , 'regionidzip' : 'int', 'yearbuilt' : 'int' , 'taxvaluedollarcnt' : 'int' , 'taxamount' : 'int' , 'censustractandblock' : 'int' , 'id.1' : 'int' , 'age' : 'int', 'month' : 'int'})
    return df

def county(fips):
  if fips == 6037:
    return 'LA'
  elif fips == 6059:
    return 'Orange'
  else:
    return 'Ventura'



def wrangle_zillow():
    cols = [
 'bathroomcnt',
 'bedroomcnt',
 # 'taxvaluedollarcnt',
 # 'logerror'
 ]
    df =  get_zillow_data()
    df = handle_missing_values(df, prop_required_column = .9, prop_required_row = .9)
    df = handle_iqr_outliers(df, trim = True, include = cols )
    df = df.drop(columns=['parcelid' , 'id' , 'calculatedbathnbr' , 'finishedsquarefeet12' , 'fullbathcnt' , 'regionidcounty' , 'roomcnt' , 'structuretaxvaluedollarcnt' , 'assessmentyear' , 'landtaxvaluedollarcnt' , 'rawcensustractandblock'])
    df = df[df.logerror < .3]
    df = df[df.logerror > -.3]
    df = df[df.taxvaluedollarcnt > 50000]
    df = df[df.taxvaluedollarcnt < 2500000]
    df = df.dropna()
    df['age'] = 2018 - df.yearbuilt
    df['month'] = pd.DatetimeIndex(df['transactiondate']).month
    df['county'] = df["fips"].apply(lambda fips: county(fips))
    df = make_int(df)
    return df



def get_iqr_outlier_bounds(df,include=None,exclude=None):
    """
    Returns dataframe with list of columns and the upper and lower bounds using the IQR method:
        LB = Q1 - 1.5 * IQR
        UB = Q3 + 1.5 * IQR
    If no columns passed to include or exclude, it defaults to finding outliers for all columns.
    Function will ignore non-numeric columns. FUTURE: Columns that contain only 0s and 1s.
    
    Returns: Pandas Dataframe 
    Parameters:
           df: dataframe in which to find outliers
      include: list of columns to find outliers for
       excude: list of columns to NOT find outliers for.  Ignored if 'include'is set.   
    
    C88
    """
    #Get Column List
    #if include and exclude are None
    if not include and not exclude:
        columns = df.columns #returns index - iterable
    elif include:
        columns = include
    else: columns = exclude
    
    #Only pull out numeric columns
    columns = df[columns].select_dtypes(include='number')
#     #TO DO: check if only 0s and 1s
#     for c in columns:
#         #if series contains only zeros and ones
#         if df[c].isin([0,1]).all():
    
    #create df for bounds
    bounds = pd.DataFrame()
    #for each column, 
    for col in columns:
        #find bounds
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3 - q1
        lb = q1 - (2* iqr)
        ub = q3 + (2 * iqr)
        #put info in df
        bounds.loc['lb',col] = lb
        bounds.loc['ub',col] = ub
    return bounds

def trim_iqr_outliers(df,bounds):
    """
    Takes in the dataset dataframe, and dataframe of the bounds for each column to be trimmed.
    Returns: Trimmed dataframe
    """
    #loop over columns to work on
    for col in bounds.columns:
        #for each col, grab the outliers
        lb = bounds.loc['lb',col]
        ub = bounds.loc['ub',col]
        #create smaller df of only rows where column is in bounds
        df = df[(df[col] >= lb) & (df[col]<=ub)]
    return df

def calc_outliers(x,lb,ub):
    ''''
    Given a value, determines if it is between the provided upper and lower bounds.  
    If b/w bounds, returns 0, else returns the distance outside of the bounds
    '''
    #if not an outlier, set to zero
    if lb <= x <= ub: return 0
    elif x < lb: return x-lb
    else: return x-ub

def add_outlier_columns(df,bounds):
    #loop over columns in bounds
    for col in bounds.columns:
        #new column name
        col_name = col + '_outlier'
        #for each column, apply the outlier calculation and store to new column
        df[col_name] = df[col].apply(calc_outliers,args=(bounds.loc['lb',col],bounds.loc['ub',col]))
    return df

##TIME PERMITTING
# Investigate why python is updating my dataframe external to the function
# needed to use df = old_df.copy() to prevent this
def handle_iqr_outliers(old_df,trim=False,include=None,exclude=None):
    """
    Takes in a dataframe and either trims outliers or creates column identifying outliers. 
    
    Outputs: None
    Returns: Pandas Dataframe
    Parameters:
                   df: dataframe in which to find outliers
                 trim: If True, will trim out any rows that contain any outliers.  
                        If False, creates new columns to indicate if row is an outlier or not.
                        Default: False
      include/exclude: list of columns to include or exclude for this function.  
                       Default is all, exclude will be ignored if include is provided.
    """    
    df= old_df.copy()    
    #Get bounds dataframe
    bounds = get_iqr_outlier_bounds(df,include,exclude)

    #If we want new column 
    if trim:
        #Function trims row if value not w/i bounds
        df = trim_iqr_outliers(df,bounds)
    else:
        #function determines if outlier and adds new columnadds new columns 
        df = add_outlier_columns(df,bounds)
        
    return df
##################################
##################################








def scale_X(train,validate,test,columns_to_scale,kind='minmax'):
    '''
    Takes prepped tr, test, validate zillow subsets. Scales the list of columns provided\
      returns dataframes concated <scaled><unscaled>.  
      
    Returns: 3 Pandas DataFrames (Train, Test, Validate)
    Inputs:
         (R) X_tr: train dataset
         (R) X_te: test dataset
        (R) X_val: validate dataset
      (R) columns: List of columns to be scaled
      (O-kw) kind: Type of scaler you want to use.  Default: minmax
                Options: minmax, standard, robust
    '''
    #Set the scaler 
    if kind.lower() == 'minmax':
        scaler = MinMaxScaler()
    elif kind.lower() == 'standard':
        scaler = StandardScaler()
    elif kind.lower() == 'robust':
        scaler = RobustScaler()
    else:
        print(f'Invalid entry for "kind", default MinMax scaler used')
        scaler = MinMaxScaler()

    #fit scaler and transform on train - needs to be stored as pd.DF in order to concat
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr[columns]),columns=columns,index=X_tr.index)
    #transform the rest
    X_te_scaled = pd.DataFrame(scaler.transform(X_te[columns]),columns=columns,index=X_te.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val[columns]),columns=columns,index=X_val.index)

    #rebuild the dataframes <scaled><unscaled>
    X_tr_scaled = pd.concat([X_tr_scaled,X_tr.drop(columns=columns)],axis=1)
    X_te_scaled = pd.concat([X_te_scaled,X_te.drop(columns=columns)],axis=1)
    X_val_scaled = pd.concat([X_val_scaled,X_val.drop(columns=columns)],axis=1)
    
    #return dataframes with scaled data
    return X_tr_scaled, X_te_scaled, X_val_scaled

