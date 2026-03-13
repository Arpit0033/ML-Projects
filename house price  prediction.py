import numpy as np
from sklearn.linear_model import LinearRegression

# House area (sq ft)
X = np.array([[500], [800], [1000], [1200], [1500]])

# House prices
y = np.array([1000000, 1500000, 2000000, 2500000, 3000000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for new house
area = [[1100]]
predicted_price = model.predict(area)

print("Predicted House Price:", predicted_price[0])