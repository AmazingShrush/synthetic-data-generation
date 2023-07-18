from statistics import mode
import pandas as pd 
import numpy as np


def date_discretisation(date):
    if not pd.isna(date):
        # Do not use "-" as connector, because Pandas might read it as date or time!!
        return str(date.quarter) + "&" + str(date.dayofweek)
    else:
        return "Na"


# Read csv file as DataFrame, and drop ROW_ID column
def read_csv_no_rowid(file_path):
    df = pd.read_csv(file_path)
    df.drop(['row_id'], axis=1, inplace=True)

    return df


# check NaN value
def nan_count(df):
    print("Total columns: " + str(len(df.columns)))
    print("Total rows: " + str(len(df)))
    print("--------------")
    print(df.isnull().sum())

# calculate the time delta
def time_process(df, early_col_name, late_col_name, second_early_col_name=None):
    '''
    If first_early_col_name exist, then use late_col - first_early_col_name, 
        else, use then use late_col - second_early_col_name, else set result as NaN
    The result is the time delta, save it as the late column
    '''
    # basic date exist
    if (pd.isna(df[early_col_name]) == False) & (pd.isna(df[late_col_name]) == False):
        return abs(df[late_col_name] - df[early_col_name]).total_seconds()
    # basic date is not exist, use the second basic date
    elif (pd.isna(second_early_col_name) == False) & (pd.isna(df[late_col_name]) == False):
        return abs(df[late_col_name] - df[second_early_col_name]).total_seconds()
    # current date is not exist
    else:
        return np.NaN


# Discretisation
def data_discretisation(column, n):
    return pd.cut(column, n).astype(str)



# Train and choose models
from sdv.lite import TabularPreset
from sdv.tabular import GaussianCopula
from sdv.tabular import CTGAN
from sdv.tabular import CopulaGAN
from sdv.tabular import TVAE
from sdv.evaluation import evaluate 

def train_tabular_model(constraints, train_data):
    print("Tabular Preset")
    tabular_model = TabularPreset(name='FAST_ML', constraints=constraints)
    tabular_model.fit(train_data)
    return tabular_model

def train_gaussiancopula_model(constraints, train_data):
    print("Gaussian Copula")
    gaussian_model = GaussianCopula(constraints=constraints)
    gaussian_model.fit(train_data)
    return gaussian_model

def train_ctgan_model(constraints, train_data):
    print("CTGAN")
    ctgan_model = CTGAN(constraints=constraints, cuda=True)
    ctgan_model.fit(train_data)
    return ctgan_model

def train_copulagan_model(constraints, train_data):
    print("CopulaGAN")
    copulagan_model = CopulaGAN(constraints=constraints, cuda=True)
    copulagan_model.fit(train_data)
    return copulagan_model

def train_tvae_model(constraints, train_data):
    print("TVAE")
    tvae_model = TVAE(constraints=constraints, cuda=True)
    tvae_model.fit(train_data)
    return tvae_model

def model_evaluation(sample, realdata):
    return evaluate(sample, realdata, metrics=['CSTest', 'KSTest', 'ContinuousKLDivergence', 'DiscreteKLDivergence'])

def evaluate_models(constraints, train_data):
    score_dict = {}

    print("Strat training ...")
    tabular_sample = train_tabular_model(constraints, train_data).sample(len(train_data))
    gaussioncopula_sample = train_gaussiancopula_model(constraints, train_data).sample(len(train_data))
    ctgan_sample = train_ctgan_model(constraints, train_data).sample(len(train_data))
    copulagan_sample = train_copulagan_model(constraints, train_data).sample(len(train_data))
    tvae_sample = train_tvae_model(constraints, train_data).sample(len(train_data))
    print("Training finished!")

    print("Strat evaluating ...")
    score_dict['tabular'] = model_evaluation(tabular_sample, train_data)
    score_dict['gaussiancopula'] = model_evaluation(gaussioncopula_sample, train_data)
    score_dict['ctgan'] = model_evaluation(ctgan_sample, train_data)
    score_dict['copulagan'] = model_evaluation(copulagan_sample, train_data)
    score_dict['tvae'] = model_evaluation(tvae_sample, train_data)
    print("Evaluating finished!")

    return  sorted(score_dict.items(), key=lambda item: item[1]).pop()[0]

    
def build_model(constraints, train_data):
    # Get the name of best model, re-fit the model again
    best_model_name = evaluate_models(constraints, train_data)
    if best_model_name == 'tabular':
        best_model = train_tabular_model(constraints, train_data)
    elif best_model_name == 'gaussiancopula':
        best_model = train_gaussiancopula_model(constraints, train_data)
    elif best_model_name == 'copulagan':
        best_model = train_gaussiancopula_model(constraints, train_data)
    elif best_model_name == 'tvae':
        best_model = train_gaussiancopula_model(constraints, train_data)
    else:
        best_model = train_ctgan_model(constraints, train_data)
    
    # Evaluating
    sample = best_model.sample(len(train_data))
    kl_continuous_score = evaluate(sample, train_data, metrics=['ContinuousKLDivergence'])
    kl_discrete_score = evaluate(sample, train_data, metrics=['DiscreteKLDivergence'])
    total_score = evaluate(sample, train_data)

    print("The best model is: " + best_model_name)
    print("The ContinuousKL_score is: " + str(kl_continuous_score))
    print("The DiscreteKL_score is: " + str(kl_discrete_score))
    print("The total score is: " + str(total_score))

    return best_model


# Save and Load model
import cloudpickle

def save_model(model, date_save_path):
    with open(date_save_path, 'wb') as f:
        cloudpickle.dump(model, f)


def load_model(date_load_path):
    with open(date_load_path, 'rb') as f:
        model = cloudpickle.load(f)
    return model


import uuid
def uuid_generate(n):
    uuid_list = []
    for i in range(n):
        uuid_list.append(str(uuid.uuid4()))
    return uuid_list


import names
def name_generate(n):
    names_list=[]
    for i in range(n):
        names_list.append(names.get_full_name())
    return names_list