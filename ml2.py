#simple linear regression w/ predicted and exprected values

import pandas as pd
from sklearn.model_selection import train_test_split

#not expected to be very accurate
nyc = pd.read_csv('ave_hi_nyc_jan_1895-2018.csv') #produces a dataframe

print(nyc.head(3)) 

#date column = data, temp = target
#need to seperate out date and temp

#each column in pandas dataframe = series (1D)

#print(nyc.Date.values) #1 Dimensional, but train/test/split expects it to be 2D, needs to be seperated into rows

print(nyc.Date.values.reshape(-1, 1)) #reshape (-1,1) = 1st arguement = number of rows, 2nd arguement = columns . -1 = create appropiate number of rows based on the data

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1,1), nyc.Temperature.values,
    random_state = 11 #controlled random. force model to randomize the data in a certain way
    #1st arguement = data, 2nd agrument = target, can be 1D
)

print(X_train.shape) #75% of our data (rows)
print(X_test.shape) #25% of samples for testing 

#training the model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X=X_train, y= y_train) #this is where the learning is taking place. 2 arguements: data, target
#y = mx+b

print(lr.coef_)#m
print(lr.intercept_)#b

predicted = lr.predict(X_test) #don't need to give it y_test b/c that's the purpose of the model to compare the predicted values to the actual ones

expected = y_test

for p,e in zip(predicted[::5], expected[::5]): #[::5] = look at every 5th element
    print(f"predicted: {p:.2f}, expected: {e: .2f}")

#have only 1 feature (year and temp) which is why predicted and expected aren't close

#predicting future years using lambda so we give it X, so it'll give us the corresponding value of y
predict = list(lambda x: lr.coef_*x + lr.intercept_)

