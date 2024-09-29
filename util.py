#### This is common utility, eg. reading files, and code backups

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline


import kagglehub

# kagglehub.login()


def data_download(input_path,version=1):
    output_path = kagglehub.dataset_download(input_path+'/versions/'+str(version))
    print('data downloaded to:',output_path)
    return output_path


def read_csv(path):

    if '.csv.gz' in path:
        return pd.read_csv(path, compression='gzip')
    else:
        return pd.read_csv(path)


def read_folder(input_path,keywords = None):
    ## check is input_path exist, ignore

    files = os.listdir(input_path)
    df_list = []
    for file_name in files:
        if (keywords is None) or (keywords in file_name):
            df = read_csv(os.path.join(input_path, file_name))
            df_list.append(df)
    
    if len(df_list)>0:
        return pd.concat(df_list)
    else:
        return pd.DataFrame()
    

def create_folder(path):
    os.makedirs(path,exist_ok=True)
    print('folder created successfully:', path)

    
#### check uniuqe categorical values
def check_unique(data_df):
    for col in data_df.columns:
        if data_df[col].dtype == 'object':
            if data_df[col].nunique() < 10:
                print('"'+col+'" has', data_df[col].nunique(),"unique values:\n", data_df[col].unique(),'\n')


### drop unnessary columns:
def drop_col(df, cols_list):
    # defined_cols = [col for col in df.columns if col in cols_list]
    # df.drop(defined_cols, axis=1, inplace=True) 
    df.drop(cols_list, axis=1, inplace=True, errors='ignore')   ## only drop exist columns


### violin plots
def visualize_features_vs_target_label(df_data, label, feature_list, n_cols=3):
    
    if len(feature_list) % n_cols == 0:
    
        number_of_rows = int(len(feature_list)/n_cols)
    else:
        number_of_rows = int(len(feature_list)/n_cols) +1
    
    fig = make_subplots(rows=number_of_rows, cols=n_cols)
    
    row_pos = 1
    col_pos = 1
    
    for feature_col in feature_list:

        fig.add_trace(
            go.Violin(x=df_data[label], y=df_data[feature_col], name=feature_col),
            row=row_pos, col=col_pos
        )
        col_pos = col_pos + 1
        
        if col_pos > n_cols:
            col_pos = 1
            row_pos = row_pos + 1

    fig.update_layout(#violingap=0, 
                  #violinmode='overlay', 
                  title_text="Features-to-Target Relations")

    fig.show()