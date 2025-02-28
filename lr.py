from parse import db
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#split 'Condition' field from dataframe for train test split
X = db.drop('Condition', axis=1)
y = db['Condition']

#split data into training data & testing data w/ a ratio of 7:3 respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

#standardize tts data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lmodel = LogisticRegression()
lmodel.fit(X_train, y_train)
y_pred = lmodel.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

#confusion matrix; visualisation for performance evaluation

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()
