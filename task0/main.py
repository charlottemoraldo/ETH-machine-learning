############################### Import libraries ##############################
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


########################## Read and extract data ##############################
train_read=pd.read_csv('train.csv', delimiter = ',')
x_train=train_read[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
y_train=train_read[['y']]

test_read=pd.read_csv('test.csv', delimiter = ',')
x_test=test_read[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]


############################ Compute estimate #################################
regr = LinearRegression();
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# Print data
print(y_pred)
print('Coefficients: \n', regr.coef_)

# Compute RMSE with mean
y_mean = np.mean(x_test,axis=1)
RMSE = mean_squared_error(y_mean, y_pred)**0.5
print('RMSE: \n', RMSE)


########################### Export to CSV file ################################
y=np.array(y_pred)
y2=y.ravel()
Id=np.arange(10000,12000,1)
Id=np.array(Id)
df=pd.DataFrame({"Id": Id, "y": y2})
df.to_csv("results.csv", index=False)
final=pd.read_csv('results.csv')
final.head()