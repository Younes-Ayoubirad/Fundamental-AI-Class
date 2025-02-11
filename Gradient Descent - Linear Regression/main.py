import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# ------------------- Data Reading
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# ------------------- Data Preprocess
x_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values

x_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

# ------------------- Scale
def min_max_scale(X, min_val=None, max_val=None):
    if min_val is None:
        min_val = X.min(axis=0)
    if max_val is None:
        max_val = X.max(axis=0)
    X_scaled = (X - min_val) / (max_val - min_val)
    return X_scaled, min_val, max_val


x_train, x_train_min, x_train_max = min_max_scale(x_train)
x_test = (x_test - x_train_min) / (x_train_max - x_train_min)

# ------------------ bias
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))


# ------------------ stochastic Gradient Descent
def stochasticGradientDescent(X, Y, parameters, learning_rate, final_learning_rate, epochs, alpha):
    num_samples = len(X)
    mse_history = []
    learning_rate_history = [learning_rate]
    previous_mse = 0
    step = (learning_rate - final_learning_rate) / epochs
    for epoch in range(epochs):
        for i in range(num_samples):
            xi = X[i:i + 1]
            yi = Y[i:i + 1]
            gradient = np.dot(xi.T, np.dot(xi, parameters) - yi)
            r1 = alpha * parameters
            gradient += r1
            parameters = parameters - learning_rate * gradient
        if epoch % 1 == 0:
            learning_rate = learning_rate - step
            learning_rate_history.append(learning_rate)
            mse = metrics.mean_squared_error(Y, np.dot(X, parameters))
            mse_history.append(mse)
            print(f"Epoch {epoch}, MSE: {mse}")
            distance = abs(mse - previous_mse)
            if distance < 0.00000005:
                return learning_rate_history, mse_history, parameters
            previous_mse = mse
    return learning_rate_history, mse_history, parameters


learning_rate = 0.01
final_learning_rate = 0.0001
epochs = 10
alpha = 0.001
step = (learning_rate - final_learning_rate) / epochs
np.random.seed(100)
parameters = np.zeros(x_train.shape[1])
learning_rate_history, mse_history, out_parameters = stochasticGradientDescent(x_train, y_train, parameters,
learning_rate=learning_rate, final_learning_rate=final_learning_rate, epochs=epochs,alpha=alpha)


mse_smooth = pd.Series(mse_history).rolling(window=5, min_periods=1).mean()

# Plot MSE over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(len(mse_history)), mse_smooth, label='MSE over epochs (smoothed)', color='blue')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE over Epochs')
plt.grid(True)
plt.legend()
plt.show()

# Plot Learning Rate over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(len(learning_rate_history)), learning_rate_history, label='Learning Rate over epochs', color='green')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')
plt.grid(True)
plt.legend()
plt.show()

# Display metrics
y_pred = np.dot(x_test, out_parameters)
r2 = metrics.r2_score(y_test, y_pred)
m1 = metrics.mean_absolute_error(y_test, y_pred)
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {m1}")

