import pandas as pd
from data_cleaning import clean_text

from utils import path_to_isot_dir, path_to_processed_dir
from sklearn.utils import shuffle

def merge_isot_df():
    true_df = pd.read_csv(path_to_isot_dir() + 'True.csv', usecols=['text'])
    fake_df = pd.read_csv(path_to_isot_dir() + 'Fake.csv', usecols=['text'])

    true_df['label'] = 1
    fake_df['label'] = 0

    merged_df = pd.concat([fake_df, true_df], ignore_index=True)
    merged_df = merged_df.rename(columns={'text':'statement'})
    merged_df = shuffle(merged_df)
    merged_df['statement'] = merged_df['statement'].str.replace(r'^[^-]*-\s', '', regex=True)

    return merged_df

def isot_clean_data_to_csv() -> None:
    df = merge_isot_df()
    df = df.dropna()
    df = df.reset_index(drop=True)

    df['statement'] = df['statement'].apply(clean_text)

    df.to_csv(path_to_processed_dir() + 'merged_isot_clean.csv', index=False)
    print('done')


def load_isot_clean_csv(nrows = None):
    df = pd.read_csv(path_to_processed_dir() + 'merged_isot_clean.csv', nrows = nrows )
    return df

# isot_clean_data_to_csv()