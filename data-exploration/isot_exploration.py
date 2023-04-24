import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.getcwd()+'\\')
sys.path.append(os.getcwd()+'\\models')
from models.isot_preprocess import load_isot_clean_csv
from models.utils import autolabel

df = load_isot_clean_csv()

true_vals = len(df[df['label'] == 1])
false_vals = len(df[df['label'] == 0])

label_counts = df['label'].value_counts()
total_samples = label_counts.sum()

labels = ['True', 'False']
counts = [true_vals, false_vals]
fig, ax = plt.subplots()

rects = ax.bar(labels, counts, color=['green', 'red'])
autolabel(rects,total_samples,ax)
for i, v in enumerate([true_vals, false_vals]):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight = 'bold')


plt.xlabel('Label')
plt.ylabel('Count')
plt.title(f'Distribution of True and False Statements ({df.shape[0]} samples)')
plt.show()

