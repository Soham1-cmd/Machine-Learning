import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
dataset = pd.read_csv("/Users/soham/Desktop/Machine Learning/iris.csv")
#print(dataset)
db = dataset.head()
print(db)
db1 = dataset.tail
print(db1)

x = dataset.iloc[:,1:3]
y = dataset['species'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier =  SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
#print(classifier)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
ac = accuracy_score(y_test,y_pred) * 100
print("Confusion matrix")
print(cm)

print("Accuray score")
print(ac)

# Plotting the decision boundary
def plot_decision_boundary(X, y, classifier, label_encoder):
    # Define bounds of the domain
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Predict class labels for every point in the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap=plt.cm.coolwarm)
    
    # Create a legend
    handles, _ = scatter.legend_elements(prop="colors")
    labels = label_encoder.inverse_transform([0, 1, 2])
    plt.legend(handles, labels, title="Species")
    
    plt.xlabel('Sepal Length (standardized)')
    plt.ylabel('Sepal Width (standardized)')
    plt.title('SVM Decision Boundary with Linear Kernel')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X_train, y_train, classifier, label_encoder)