#### This is common utility, eg. reading files, and code backups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import kagglehub

kagglehub.login()


def data_download(input_path,version=1):
    output_path = kagglehub.dataset_download(input_path+'/versions/'+str(version))
    print('data downloaded to:',output_path)
    return output_path


def read_csv(path,file=None):
    if '.csv.gz' in file:
        return pd.read_csv(path+'/'+file, compression='gzip')
    else:
        return pd.read_csv(path+'/'+file)


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

    


