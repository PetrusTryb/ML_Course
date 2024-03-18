import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
print("got data")
#inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]
obs_matrix = np.column_stack((np.ones(x_train.shape), x_train))
theta_best = np.linalg.inv(obs_matrix.T.dot(obs_matrix)).dot(obs_matrix.T).dot(y_train)
print('Theta:', theta_best)

# TODO: calculate error
mse = np.sum(((theta_best[0] + theta_best[1] * x_test) - y_test) ** 2) / len(y_test)
print('MSE:', mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y, color='red', label=f'y = {theta_best[0]} + {theta_best[1]}x')
plt.legend(loc='upper right', fontsize='small')
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Closed-form solution')
plt.show()

# TODO: standardization
x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train_standarized = (x_train - x_train_mean) / x_train_std

y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
y_train_standarized = (y_train - y_train_mean) / y_train_std

# TODO: calculate theta using Batch Gradient Descent
obs_matrix = np.column_stack((np.ones(x_train.shape), x_train_standarized))
theta_best = np.random.rand(2)
n = len(y_train)
gradients = [1, 1]
learning_rate = 0.01
epochs = 1000
for i in range(epochs):
	gradients = 2/n * obs_matrix.T.dot(obs_matrix.dot(theta_best) - y_train_standarized)
	theta_best = theta_best - learning_rate * gradients
	if i%100==0:
		mse = np.sum(((theta_best[0] + theta_best[1] * x_train_standarized) - y_train_standarized) ** 2) / len(y_train)
		print('MSE:',mse)
print('Theta:', theta_best)

# TODO: calculate error
x_test_standarized = (x_test - x_train_mean) / x_train_std
y_test_pred = float(theta_best[0]) + float(theta_best[1]) * x_test_standarized
y_test_rest = y_test_pred * y_train_std + y_train_mean
mse = np.sum((y_test_rest - y_test) ** 2) / len(y_test)
print('MSE:', mse)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * ((x-x_train_mean)/x_train_std)
y = y*y_train_std + y_train_mean
plt.plot(x, y, color='red', label=f'y = {theta_best[0]} + {theta_best[1]}x')
plt.legend(loc='upper right', fontsize='small')
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Batch Gradient Descent')
plt.show()