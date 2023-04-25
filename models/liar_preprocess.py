import pandas as pd
from sklearn.model_selection import train_test_split
from utils import path_to_liar_dir, path_to_processed_dir
from data_cleaning import clean_text

def liar_clean_data(path: str) -> pd.DataFrame:

    df = pd.read_csv(path_to_liar_dir() + path, sep='\t', header=None)
    
    df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true',
                            'false', 'half_true', 'mostly_true', 'pants_on_fire', 'location']

    label_dict = {'pants-fire': 0, 'FALSE': 0, 'barely-true': 0, 'half-true': 1,
                  'mostly-true': 1,'TRUE': 1}
    
    df['label'] = df['label'].map(label_dict)
    df = df.dropna()
    df = df.reset_index(drop=True)

    df['statement'] = df['statement'].apply(clean_text)

    df.to_csv(path_to_processed_dir() + 'merged_liar_clean.csv', index=False)
    print('done')

def load_liar_clean_csv(nrows = None):
    df = pd.read_csv(path_to_processed_dir() + 'merged_liar_clean.csv',nrows = nrows)
    return df

# liar_clean_data('merged.tsv')