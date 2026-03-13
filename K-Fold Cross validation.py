from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

model = LinearRegression()

kfold = KFold(n_splits=5)

scores = cross_val_score(model,X,y,cv=kfold)

print("Cross Validation Scores:",scores)
print("Average Score:",scores.mean())