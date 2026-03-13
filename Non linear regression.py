import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [40, 50, 55, 65, 75]
}

df = pd.DataFrame(data)

# Polynomial Features
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(df[['Hours']])

# Model training
model = LinearRegression()
model.fit(X_poly, df['Marks'])

# Prediction for 6 hours
prediction = model.predict(poly.transform([[6]]))

print("Predicted Marks:", prediction[0])