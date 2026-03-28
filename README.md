# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary libraries.
2. Load the dataset.
3. Preprocess the data (handle missing values, encode categorical variables).
4. Split the data into features (X) and target (y).
5. Divide the data into training and testing sets. 6.Create an SGD Regressor model. 7.Fit the model on the training data. 8.Evaluate the model performance. 9.Make predictions and visualize the results.

   
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

<img width="647" height="498" alt="image" src="https://github.com/user-attachments/assets/9cf9f005-a32e-4682-a3b5-20a1c3e709bb" />
<img width="395" height="576" alt="image" src="https://github.com/user-attachments/assets/8c315d12-4648-4244-af46-10b9ac7eeebb" />
<img width="672" height="281" alt="image" src="https://github.com/user-attachments/assets/ef3d2d28-2fbf-4129-8f4c-02059500c9be" />
<img width="565" height="453" alt="image" src="https://github.com/user-attachments/assets/f666e4fb-634b-4e2f-b2dd-0a2de06c8e1c" />

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
