''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = datasets.load_diabetes()


#how many sameples (rows) and How many features (columns/targets)?
print(dataset.data.shape) #442 samples/rows and 10 features/columns

# What does feature s6 represent?
#print(dataset.DESCR) #blood sugar level


#print out the coefficient

#age = target, data = all other columns??
diabetes = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#print(diabetes)



X = diabetes[['sex', 'bmi', 'bp', 's1', 's2', 's3', 's4','s5','s6']]
Y = diabetes['age']

X_train, X_test, y_train, y_test = train_test_split(X, Y,
    random_state = 11)

print(X_train.shape) 
print(X_test.shape)

lr = LinearRegression()

lr.fit(X=X_train, y= y_train) #this is where the learning is taking place. 2 arguements: data, target
#y = m(coefficient)x+b (intercept)

print(lr.coef_)#m

#print out the intercept

print(lr.intercept_)

# create a scatterplot with regression line

predicted = lr.predict(X_test) 

expected = y_test
#above = correct

for p,e in zip(predicted[::5], expected[::5]): 
    print(f"predicted: {p:.2f}, expected: {e: .2f}")

predict = lambda x1, x2, x3,x4,x5,x6,x7,x8,x9: lr.coef_[0]*x1 + lr.coef_[1]*x2+lr.coef_[2]*x3+lr.coef_[3]*x4+lr.coef_[4]*x5+lr.coef_[5]*x6+lr.coef_[6]*x7+lr.coef_[7]*x8+lr.coef_[8]*x9+lr.intercept_

import seaborn as sns


axes = sns.scatterplot(
    data=diabetes,
    #x = ['sex', 'bmi', 'bp', 's1', 's2', 's3', 's4','s5','s6'],
    y = 'age',
    #hue= '',
    palette ='winter',
    legend = False,
)

axes.set_ylim(10,70)

import numpy as np
x = np.array([min(diabetes.values), max(diabetes.values)]) #creating regression line. need 4 points: beginning points of x and y and end points of x and y
print(x)
y = predict(x)
print(y)

#creating the line
import matplotlib.pyplot as plt

line = plt.plot(x,y)
plt.show()
