import pandas as pd
import numpy as np
import os
# import sklearn



def path_to_liar_dir() -> str:
    return os.getcwd() + '\\data\\liar-dataset\\'

def path_to_isot_dir() -> str:
     return os.getcwd() + '\\data\\ISOT-News-dataset\\'

# path for your data

FAKE_ISOT = path_to_isot_dir() + "Fake.csv"
LIAR_TEST = path_to_liar_dir() + 'test.tsv'

def tsv_reader(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t')

# ISOT Data Set
# df_ISOT = pd.read_csv(FAKE_ISOT)

# LIAR Data Set
df_LIAR = tsv_reader(LIAR_TEST)
df_LIAR.columns = ["id",
                "label",
                "statement",
                "subject",
                "speaker",
                "job_title",
                "state_info",
                "party_affiliation",
                "barely_true_counts",
                "false_counts",
                "half_true_counts",
                "mostly_true_counts",
                "pants_on_fire_counts",
                "context"
                ]

# print(df_LIAR.head())
# print(df_LIAR.info())
# print(df_LIAR.shape)
# print(df_ISOT.head()['text'])
# print(df_ISOT.astype())


