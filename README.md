# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset and display basic information.
2.Preprocess the data by removing unnecessary columns and converting categorical variables.
3.Split the dataset into input features (X) and target variable (y).
4.Standardize the feature set and target values using scaling techniques.
5.Divide the data into training and testing sets.
6.Train the model using SGD Regressor on the training data.
7.Predict the output, evaluate performance, and visualize actual vs predicted values.
   
## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: SATHISH H
RegisterNumber:  212225240142
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

#Load the dataset
data=pd.read_csv("CarPrice_Assignment (4).csv")
print(data.head())
print(data.info())

#Data Processing
#Dropping unnecessary columns and handling categorical variables
data=data.drop(['CarName','car_ID'], axis=1)
data=pd.get_dummies(data,drop_first=True)

#Splitting the data into features and target variables
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler = StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predictions
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

#Print evaluation metrics
print('Name: SATHISH H')
print('Reg. No: 212225240142')
print("Mean Squared Error:",mse)
print("R-squared Score:",r2)
print("Mean Absolute Error:",mae)

#Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red') #Perfect prediction line
plt.show()


```

## Output:

<img width="500" height="369" alt="image" src="https://github.com/user-attachments/assets/ad64e3dc-ad6b-4cd4-91c8-aaab5d0a8f28" />
<img width="316" height="418" alt="image" src="https://github.com/user-attachments/assets/da68eb39-33c9-4636-b827-914019d70b05" />
<img width="504" height="143" alt="image" src="https://github.com/user-attachments/assets/4c58af68-e07d-4956-9838-6e8ba4a1717a" />
<img width="519" height="349" alt="image" src="https://github.com/user-attachments/assets/0a691a16-53a0-497e-a284-5e56c45278b9" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
