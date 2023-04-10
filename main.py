import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
## to display in notebook:
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import and explore data
customers = pd.read_csv("Ecommerce Customers")

customers.head()
customers.info()
customers.describe()

# create jointplots to compare the Time on Website and Time on App with Yearly Amount Spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

# use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)

# find most correlated feature with Yearly Amount spent using pairplot and create a linear model plot of that feature and Yearly Amount spent
sns.pairplot(customers)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

# Train and test the data using linear regression 
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

# evaluate the model and check coefficients of the model for each feature

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test-predictions),bins=50) # check that residuals are normally distributed

coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
