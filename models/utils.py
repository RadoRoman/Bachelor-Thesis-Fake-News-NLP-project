import pandas as pd
import numpy as np
import os
# import sklearn



def path_to_liar_dir() -> str:
    return os.getcwd() + '\\data\\liar-dataset\\'

def path_to_isot_dir() -> str:
     return os.getcwd() + '\\data\\ISOT-News-dataset\\'

def path_to_processed_dir() -> str:
     return os.getcwd() + '\\data\\processed\\'

# path for your data

FAKE_ISOT = path_to_isot_dir() + "Fake.csv"
LIAR_TEST = path_to_liar_dir() + 'test.tsv'

def tsv_reader(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

def autolabel(rects, total_samples, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height/2, f'{height / total_samples:.1%}',
                ha='center', va='bottom', color='white')


