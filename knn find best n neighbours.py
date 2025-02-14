from parse import db
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt 


#split 'Condition' field from dataframe for train test split
X = db.drop('Condition', axis=1)
y = db['Condition']

#container variables for K (neighbour count for iterative test
K = [] 
traintest_scores = [] 
scores = {} 

#split data into training data & testing data w/ a ratio of 7:3 respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#standardize tts data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#iterative KNN classifier test- sampling for best quantity for n_neighbours
for k in range(2, 21): 
    knn = KNeighborsClassifier(n_neighbors = k) 
    knn.fit(X_train, y_train) 
  
    training_score = knn.score(X_train, y_train) 
    test_score = knn.score(X_test, y_test) 
    K.append(k) 
  
    traintest_scores.append((training_score, test_score)) 
    scores[k] = [training_score, test_score]
    print(scores[k])

#visualise data from iterative KNN classifier test
plt.scatter(K, [i[0] for i in traintest_scores], color ='g') 
plt.scatter(K, [i[1] for i in traintest_scores], color ='r') 
plt.show()
