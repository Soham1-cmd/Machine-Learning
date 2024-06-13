import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

dataset = pd.read_csv('/Users/soham/Desktop/Machine Learning/data1.csv')
#print(dataset)
db = dataset.shape
print(db)
db1 = dataset.head()
print(db1)
x = dataset[['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']]
y = dataset['diagnosis']
x.head()
print(x)
print(y)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test  = train_test_split(x,y,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
dT = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
dT.fit(X_train,y_train)

y_pred = dT.predict(X_test)
 
acc = accuracy_score(y_test,y_pred)
print(acc)

from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test.values,y_pred)
print(cm)

print(classification_report(y_pred,y_test))

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

labels = y.unique() 
plot_confusion_matrix(cm, labels) 



