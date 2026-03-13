import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
'Gender':['Male','Female','Male','Female'],
'Experience':[1,3,5,2],
'Salary':[20000,30000,50000,25000]
}

df = pd.DataFrame(data)

df = pd.get_dummies(df,drop_first=True)

X = df[['Experience','Gender_Male']]
y = df['Salary']

model = LinearRegression()
model.fit(X,y)

print(model.predict([[4,1]]))