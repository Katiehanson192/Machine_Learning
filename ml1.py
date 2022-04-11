from sklearn.datasets import load_digits #load_digits = a dataset, to change, just pick a different dataset

digits = load_digits()

print(digits.data[:2]) #64 numbers that represent a number
                        #possible numbers: 0-16 b/c this is the pixel intensity for each of the numbers?

print(digits.data.shape) #returns # of rows and columns, 64 columns = 8x8 dataframe?

print(digits.target[:2]) #returns [0 1] the first target. target of these 2 numbers belong to the number 0
                        #pixel intensity corresponds to all of these numbers 
                        #the possible numbers are: 0 -9
print(digits.target.shape) #returns rows (samples) and # of columns (features)
                            #target attribute has one 1 column b/c target only has one value (the pixels are representing 1 value-drawing out 1 number)
                            
'''
data array: The 1797 samples (digit images), each with 64 features 
with values 0 (white) to 16 (black), representing pixel intensities
'''

#packages to install: sklearn, pandas, matplotlib, seaborn
#each number has multiple sets of values for pixel intensity (each of the 1797 samples have their own set of values depending on their target - number that the pixels are forming)

#so far = 1 dimensional array

print(digits.images[:2]) #creates a 2D array
                        #CAN'T be used in model (model needs 1D array)

import matplotlib.pyplot as plt

fig, axes =plt.subplots(nrows=4, ncols=6, figsize = (6,4))

for item in zip(axes.ravel(), digits.images,digits.target): #axes.ravel = flatten 2D array, each graph has an image and a target on it
    axes,image,target = item #zip method = will iterate more than 1 list at the same time
                            #splitting up the objects in the for loop seperately 
    axes.imshow(image, cmap = plt.cm.gray_r) #takes image to show up in each figure and shows target of image at top
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
plt.show()


#training and testing set sizes
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, random_state=11
) #random_state for reproducibility

#this shows that training is 75%
print(x_train.shape)

#this shows testing is 25%
print(x_test.shape)

#creating the model 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X=x_train, y=y_train)

#returns an array containing the predicted class of each test image:
#creates an array of digits 
predicted = knn.predict(X=x_test) #x parameter needs to always be Capitalized!

#array of the expected digits
expected = y_test

print(predicted[:20])

print(expected[:20]) #most of the first 20 elements were predicted as expected

#locate all incorrect predictions for the entire test set
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p !=e] #if expected and predicted are not equal, add it to the "wrong" list

print(wrong)

#metrics for measuring model accuracy
print(format(knn.score(x_test, y_test), ".2%")) #print the accuracy rate for the model (comparing data test and the target test)

#confusion matrix
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)

#visualizing the confusion matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot = True, cmap= plt2.cm.nipy_spectral_r)
plt2.show()

print("done")


#Lambda functions
    #doesn't have a name, can be created at any point in the code, can only process 1 expression - can only produce 1 result
    #ex: remainder = lambda num: num % 2 --> function object. remainder becomes a function. same thing as def remainder(num): return num % 2
    #most useful when high level function, calling a lower level function