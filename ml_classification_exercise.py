# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.

#EXERCISE

'''
load the iris dataset and use classification 
to see if the expected and predicted species match up
'''
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

iris = load_iris()

'''
display the shape of the data, target and target_names
'''

print(iris.data.shape) #150 rows, 4 columns
print(iris.target.shape) 
print(iris.target_names) #species—Iris setosa, Iris versicolor and Iris virginica.
print(iris.target_names.shape) #3 

#need to create subplots for display?

'''
display the first 10 predicted and expected results using
the species names not the number (using target_names)
'''

#Splitting the data

x_train, x_test, y_train, y_test = train_test_split(
    iris.data,  iris.target, random_state=11
)
#print(x_train.shape) #112 samples used in training dataset, 4 columns
#print(x_test.shape) #38 samples used in testing dataset, 4 columns

#Creating the model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X=x_train, y= y_train) #fit = training the model

predicted = knn.predict(X=x_test) #creates array of predicted flower type for each row in test dataset
expected = y_test

print(iris.target_names[predicted[:10]])
print(iris.target_names[expected[:10]])

'''
display the values that the model got wrong
'''
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p !=e] 

print(wrong)

#metrics for measuring model accuracy
print(format(knn.score(x_test, y_test), ".2%"))

'''
visualize the data using the confusion matrix
'''
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot = True, cmap= plt2.cm.nipy_spectral_r)
plt2.show()

print("done")