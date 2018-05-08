import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv")

print(data.shape)

# The mean sale price is:
print(data['SalePrice'].mean())

# The year the oldest house was built is:
print(data['YearBuilt'].min())

# plot distribution of Iowa house SalePrice
plt.hist(data.SalePrice, bins=50)
plt.show()

# plot a histogram of the YearBuilt column
plt.hist(data.YearBuilt, bins=50)
plt.show()

# Create Correlation Matrix Heatmap
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

# Top 10 Heatmap - we can see the top 10 features associated with SalePrice
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# prediction target
y = data.SalePrice
# set predicting features
predictors = ['GarageArea', 'OverallQual', 'TotalBsmtSF', 'YearBuilt', 'GrLivArea','GarageCars', 'TotRmsAbvGrd', '1stFlrSF']
X = data[predictors]

# fit the model
model = DecisionTreeRegressor()
model.fit(X, y)
# make predictions with your model
predictions = model.predict(X)

# compare the model's predictions with the true sale prices of the first few houses
print(X.assign(Prediction = predictions).assign(Y = y).head())

# Model evaluation
# Using mean absolute error

val_model = DecisionTreeRegressor()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
val_model.fit(train_X, train_y)
val_predictions = val_model.predict(val_X)
mean_absolute = mean_absolute_error(val_y, val_predictions)
print(mean_absolute)

# make predictions for the test data here
test = pd.read_csv('test.csv')

# use Imputer to impute values
from sklearn.preprocessing import Imputer
my_imputer = Imputer()

for col in test:
    if test[col].dtype == object:
        del test[col]

imputed_test = my_imputer.fit_transform(test.values)
new_test = pd.DataFrame(imputed_test, index=test.index, columns=test.columns)

test_X = new_test[predictors]

# used to test predictions
test_predictions = model.predict(test_X)
print(test_predictions)