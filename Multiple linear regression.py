from sklearn.linear_model import LinearRegression
import pandas as pd

data = {
'Hours':[1,2,3,4,5],
'Sleep':[6,7,5,8,6],
'Marks':[30,40,50,70,80]
}

df = pd.DataFrame(data)

X = df[['Hours','Sleep']]
y = df['Marks']

model = LinearRegression()
model.fit(X,y)

print("Prediction:",model.predict([[3,7]]))