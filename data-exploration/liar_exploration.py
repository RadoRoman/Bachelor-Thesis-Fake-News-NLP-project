import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('c:\\Work\\Bachelor-Thesis-Fake-News-NLP-project\\')
from models.utils import path_to_liar_dir, autolabel


df = pd.read_csv(path_to_liar_dir() + 'merged.tsv', sep='\t', header=None)

df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party', 'barely_true',
                           'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context']
df = df[['label', 'statement']]

label_counts = df['label'].value_counts()
total_samples = label_counts.sum()

fig, ax = plt.subplots()
rects = ax.bar(label_counts.index, label_counts.values)

autolabel(rects, total_samples, ax)

for i, v in enumerate(label_counts.values):
    ax.text(i, v+5, str(v), ha='center', fontweight='bold')

ax.bar(label_counts.index, label_counts.values)
ax.set_title(f'Label Distribution ({df.shape[0]} samples)')
ax.set_xlabel('Label')
ax.set_ylabel('Count')
plt.show()
