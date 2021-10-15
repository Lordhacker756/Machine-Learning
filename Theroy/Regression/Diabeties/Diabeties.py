# from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#Importing data set for regression ie observing data and predict a value
diabeties = datasets.load_diabetes();

#To check what all is there in the data set we use
#print(diabeties.keys())

#Lets make the best fit line, for that first have the points
diabeties_x = diabeties.data #Numpy array

#Now that array has so many data points so we split it and now make our sliced array for training
diabeties_x_train = diabeties_x[:-30] #End ke 30

#now to test the trained algo we need test data that we'll take beiginning ke 20
diabeties_x_test = diabeties_x[-30:]

#Same thing for Y axis
diabeties_y_train = diabeties.target[:-30]
diabeties_y_test = diabeties.target[-30:]

#X axis pe hamare pass feature hai and y axis pe label => Feature dekh ke label bataya jata

#creating linear model
model = linear_model.LinearRegression()

#Now to create the axis with the points we got, and train the algo
model.fit(diabeties_x_train,diabeties_y_train)

#Testing the model and finding predicted values
diabetic_y_predict = model.predict(diabeties_x_test)

#Checking how accurate the predictions are Sum of squared error ka avg mean_squared_error(actual vals, predicted vals)
print("The Mean Error is ", mean_squared_error(diabeties_y_test,diabetic_y_predict))


#Let's find the weights(w1,w2,w3....) and intercept(w0)
print("Weight is: ",model.coef_)
print("Intercept is: ",model.intercept_)

print("Code Ran")
#Lets plot these
# plt.scatter(diabeties_x_test,diabeties_y_test)
# plt.plot(diabeties_x_test,diabetic_y_predict,label="Mean Square Error")
# plt.legend()
# plt.title("Diabeties")
# plt.ylabel("Predicted Values")
# plt.xlabel("Actual Values")
# plt.show()
