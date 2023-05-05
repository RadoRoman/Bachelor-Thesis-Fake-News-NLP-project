import matplotlib.pyplot as plt
import numpy as np

confusion_matrix = np.array([[1116, 0], [1084,0]])
labels = ['Fake', 'True']

fig, ax = plt.subplots()

im = ax.imshow(confusion_matrix, cmap='Blues')
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
cbar = ax.figure.colorbar(im, ax=ax)


for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

ax.set_title('Confusion Matrix for using SVM on ISOT')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('Actual Label')
plt.show()
