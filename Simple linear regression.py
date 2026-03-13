import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {'Hours':[1,2,3,4,5], 'Marks':[20,40,50,65,80]}
df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Prediction:",pred)
print("MSE:",mean_squared_error(y_test,pred))