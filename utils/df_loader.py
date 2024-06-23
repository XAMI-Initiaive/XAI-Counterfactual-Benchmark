import pandas as pd

from utils.preprocessing import get_columns_type, transform_to_dummy, label_encode, remove_missing_values

import numpy as np


def get_loading_fn(dataset_name):
    if dataset_name == 'electricity':
        dataset_loading_fn = load_electricity_df
    elif dataset_name == 'adult':
        dataset_loading_fn = load_adult_df
    elif dataset_name == 'german':
        dataset_loading_fn = load_german_df
    elif dataset_name == 'compas':
        dataset_loading_fn = load_compas_df
    elif dataset_name == 'diabetes':
        dataset_loading_fn = load_diabetes_df
    elif dataset_name == 'breast_cancer':
        dataset_loading_fn = load_breast_cancer_df
    else:
        raise Exception("Unsupported dataset")
    return dataset_loading_fn

def load_adult_df():
    ##### Pre-defined #####
    target_name = 'class'

    df = pd.read_csv('./datasets/adult.csv', delimiter=',', skipinitialspace=True)

    del df['fnlwgt']
    del df['education-num']

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_breast_cancer_df():
    
    target_name = 'diagnosis'

    df = pd.read_csv('./datasets/breast_cancer.csv',
                     delimiter=',', skipinitialspace=True)

    del df['id']
    del df['Unnamed: 32']        

    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_diabetes_df():
    
    target_name = 'Outcome'

    df = pd.read_csv('./datasets/diabetes.csv',
                     delimiter=',', skipinitialspace=True)

    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes
    
def load_german_df():
    
    target_name = 'default'

    df = pd.read_csv('./datasets/german.csv',delimiter=',', skipinitialspace=True)

    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    transform_cat_col = ['credits_this_bank', 'people_under_maintenance']
    for col in transform_cat_col:
        df[col] = df[col].apply(str)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_compas_df():
    
    target_name = 'class'

    df = pd.read_csv('./datasets/COMPAS.csv',delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'

    # We add some categorical columns manually.
    transform_cat_col = ['is_recid', 'is_violent_recid', 'two_year_recid']
    for col in transform_cat_col:
        df[col] = df[col].apply(str)

    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    feature_names = [col for col in df.columns if col != target_name]

    df = remove_missing_values(df)

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_electricity_df():
    
    target_name = 'target'

    df = pd.read_csv('./datasets/electricity.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_covertype_df():
    
    target_name = 'target'

    df = pd.read_csv('./datasets/covertype.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_pol_df():
    
    target_name = 'target'

    df = pd.read_csv('./datasets/pol.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_house_16H_df( ):
    
    dataset_name = "house_16H"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_MagicTelescope_df( ):
    
    dataset_name = "MagicTelescope"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes



def load_bank_marketing_df( ):
    
    dataset_name = "bank-marketing"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes



def load_Bioresponse_df( ):
    
    dataset_name = "Bioresponse"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_MiniBooNE_df( ):
    
    dataset_name = "MiniBooNE"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_default_of_credit_card_clients_df( ):
    
    dataset_name = "default-of-credit-card-clients"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_Higgs_df( ):
    
    dataset_name = "Higgs"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_eye_movements_df( ):
    
    dataset_name = "eye_movements"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_Diabetes130US_df( ):
    
    dataset_name = "Diabetes130US"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_jannis_df( ):
    
    dataset_name = "jannis"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes



def load_heloc_df( ):
    
    dataset_name = "heloc"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes




def load_credit_df( ):
    
    dataset_name = "credit"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes


def load_california_df( ):
    
    dataset_name = "california"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_electricity_mixed_df( ):
    
    dataset_name = "electricity_mixed"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_eye_movements_mixed_df( ):
    
    dataset_name = "eye_movements_mixed"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_covertype_mixed_df( ):
    
    dataset_name = "covertype_mixed"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_albert_df( ):
    
    dataset_name = "albert"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes

def load_road_safety_df( ):
    
    dataset_name = "road-safety"
    target_name = 'target'

    df = pd.read_csv(f'./datasets/{dataset_name}.csv',delimiter=',')
    
    df[target_name] = df[target_name].apply(lambda x: "Y" if x==1 else "N")
    
    feature_names = [col for col in df.columns if col != target_name]
    # delete target_name from feature_names

    possible_outcomes = list(df[target_name].unique())

    numerical_cols, categorical_cols, columns_type = get_columns_type(df)

    numerical_cols =  [col for col in numerical_cols if col != target_name]

    categorical_cols = [target_name]
    
    return df, feature_names, numerical_cols, categorical_cols, columns_type, target_name, possible_outcomes
