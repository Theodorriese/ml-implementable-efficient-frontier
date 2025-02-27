import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Settings ---
def f(x):
    return 10 * x + 0.5 * x**2 - 0.01 * x**3 - 1.08**x + 100 * np.sin(x)

np.random.seed(1)

# Data
n = 1000
x = np.random.uniform(-30, 70, n)
y_true = f(x)
y = y_true + np.random.normal(0, 150, n)

# True function and OLS approximation ----
x_seq = np.linspace(-30, 70, 101)
y_true_seq = f(x_seq)
y_noisy = y_true_seq + np.random.normal(0, 150, len(x_seq))

# Fit a linear regression model
x_seq_reshaped = x_seq.reshape(-1, 1)
regressor = LinearRegression().fit(x_seq_reshaped, y_true_seq)
y_pred = regressor.predict(x_seq_reshaped)

# Plot
plt.scatter(x_seq, y_noisy, label="Noisy Data", color="gray", alpha=0.5)
plt.plot(x_seq, y_true_seq, label="True Function", color="blue")
plt.plot(x_seq, y_pred, label="OLS Approximation", color="red")
plt.legend()
plt.title("True Function and OLS Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Various RFF approximations ------------

# Settings
np.random.seed(1)
ps = np.arange(2**1, 2**9 + 1, 2)  # Sequence from 2^1 to 2^9
lambdas = np.exp(np.arange(0, 11))  # Exponentially increasing lambda values

# Generate data (replace n and x with your existing data if available)
n = 1000
x = np.random.uniform(-30, 70, size=n)
y_true = 10 * x + 0.5 * x**2 - 0.01 * x**3 - 1.08**x + 100 * np.sin(x)
y = y_true + np.random.normal(0, 150, size=n)

# Train/test split
train_idx, test_idx = train_test_split(np.arange(n), test_size=0.5, random_state=1)
x_train, x_test = x[train_idx].reshape(-1, 1), x[test_idx].reshape(-1, 1)
y_train, y_test = y[train_idx], y[test_idx]
y_true_train, y_true_test = y_true[train_idx], y_true[test_idx]

# Predict multiple lambdas and collect in tidy format
# Predict multiple lambdas and collect in tidy format
def pred_tidy(x_act, x, y, y_true, lambdas):
    """
    Predict multiple lambda values and collect results in a tidy DataFrame.

    Parameters:
        x_act (array-like): Actual x values.
        x (array-like): Feature matrix.
        y (array-like): Target values.
        y_true (array-like): True target values.
        lambdas (array-like): Regularization parameter values.

    Returns:
        pd.DataFrame: Tidy DataFrame containing predictions and actual values.
    """
    preds = []
    for lam in lambdas:
        ridge_model = Ridge(alpha=lam, fit_intercept=True)
        ridge_model.fit(x, y)  # Fit the model with the current lambda
        preds.append(ridge_model.predict(x))
    preds = np.column_stack(preds)  # Stack predictions for all lambdas
    log_lambdas = np.log(lambdas)
    preds_df = pd.DataFrame(preds, columns=log_lambdas)
    tidy_df = preds_df.melt(var_name="log_lambda", value_name="pred")

    # Repeat y, x_act, and y_true to match tidy_df length
    n_repeats = len(lambdas)
    tidy_df["y"] = np.tile(y, n_repeats)
    tidy_df["x"] = np.tile(x_act, n_repeats)
    tidy_df["y_true"] = np.tile(y_true, n_repeats)
    return tidy_df


# Predict on training data for multiple lambdas
train_predictions = pred_tidy(
    x_act=x[train_idx], x=x_train, y=y_train, y_true=y_true_train, lambdas=lambdas
)

# Predict on test data for multiple lambdas
test_predictions = pred_tidy(
    x_act=x[test_idx], x=x_test, y=y_test, y_true=y_true_test, lambdas=lambdas
)

# Combine training and test predictions
train_predictions["split"] = "train"
test_predictions["split"] = "test"
predictions = pd.concat([train_predictions, test_predictions], ignore_index=True)

print(predictions.head())

# Nået til SIMULATIONS
