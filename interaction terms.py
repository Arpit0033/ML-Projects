import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'Hours':[2,3,4,5,6],
    'Sleep':[6,7,5,8,7],
    'Marks':[50,55,65,70,80]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Hours','Sleep']]
y = df['Marks']

# Polynomial interaction
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_interaction = poly.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_interaction, y)

# Prediction
prediction = model.predict(poly.transform([[3,7]]))

print("Predicted Marks:", prediction[0])