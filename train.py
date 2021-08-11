import pickle

import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import seaborn as sns


def load_feature_label(file_name):
    feature_label = np.load(file_name)
    return feature_label[:, :-1], feature_label[:, -1].astype(np.uint8)

train_feature, train_label = load_feature_label('train_features.npy')
model = SVC(kernel='rbf', C=1e3, gamma=0.5, class_weight='balanced', probability=True)
model.fit(train_feature, train_label)

with open('new_liveness.pkl', 'wb') as f:
    pickle.dump(model, f)

test_feature, test_label = load_feature_label('test_features.npy')
print(test_feature)
pred = model.predict(test_feature)
class_names=['FAKE', 'REAL']
matrix = confusion_matrix(test_label, pred)
score = accuracy_score(test_label, pred)
report = classification_report(test_label, pred)
print(score)
print(matrix)
print(report)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cmap='Blues')
plt.title('Confusion matrix'), plt.tight_layout()
plt.ylabel('True Class'), plt.xlabel('Predicted Class')
plt.show()