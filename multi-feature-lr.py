import numpy as np
import csv

ITERATIONS = 50000  # Reduce iterations for efficiency
ALPHA = 0.001
TOLERANCE = 1e-6  # Stop if gradient updates are very small

x = []
y = []

# --------------- Load training data ---------------
with open('real_estate_prices.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        x.append(np.array(row[:-1], dtype=float))
        y.append(float(row[-1]))

# Convert to NumPy arrays
X_train = np.array(x)
Y_train = np.array(y)  # Target values

# --------------- Feature Scaling ---------------
def scaling(X):
    mean = np.mean(X, axis=0)  # Compute mean for each feature
    std = np.std(X, axis=0)  # Compute standard deviation for each feature
    std[std == 0] = 1  # Avoid division by zero

    X_standardized = (X - mean) / std
    return X_standardized, mean, std  # Return mean & std for later use

# Scale features
X_scaled, mean, std = scaling(X_train)

# Scale target values
y_mean = np.mean(Y_train)
y_std = np.std(Y_train)
Y_train = (Y_train - y_mean) / y_std  # Standardize target values
# -------------------------------------------------

# --------------- Hypothesis function ---------------
def f(X, w, b): 
    return np.dot(X, w) + b  
# ---------------------------------------------------

# --------------- Cost function (Mean Squared Error) ---------------
def c(X, y, w, b):
    return np.sum((f(X, w, b) - y) ** 2) / (2 * len(y))
# ------------------------------------------------------------------

# --------------- Compute Gradients ---------------
def calc_grad(X, y, w, b):
    m = len(y)
    err = f(X, w, b) - y  # Error

    dj_dw = np.dot(X.T, err) / m
    dj_db = np.mean(err)

    return dj_dw, dj_db
# -------------------------------------------------

# --------------- Gradient Descent with Early Stopping ---------------
def grad_desc(X, y, w, b):
    cost_history = []  # Store cost values for debugging

    for i in range(ITERATIONS):
        dj_dw, dj_db = calc_grad(X, y, w, b)

        # Early stopping if gradients are too small
        if np.linalg.norm(dj_dw) < TOLERANCE and abs(dj_db) < TOLERANCE:
            print(f"Stopping early at iteration {i} due to small updates")
            break

        w -= ALPHA * dj_dw
        b -= ALPHA * dj_db

        # Store cost
        cost = c(X, y, w, b)
        cost_history.append(cost)

        # Print cost every 10% of iterations
        if i % (ITERATIONS // 10) == 0:
            print(f"Iteration {i:6d}: Cost {cost:.6f}")

    return w, b, cost_history
# ------------------------------------------------

# Initialize weights and bias
m = X_train.shape[1]  # Number of features
w = np.zeros(m)  # Initialize weights
b = 0  # Initialize bias

# Train the model
w, b, cost_history = grad_desc(X_scaled, Y_train, w, b)

# Print final weights and bias
print("\nOptimized Weights:", w)
print("Optimized Bias:", b)

# --------------- Undo Scaling for Weights & Bias ---------------
def unscale_y(scaled_y, y_mean, y_std):
    return scaled_y * y_std + y_mean

# Convert scaled weights back to original scale
w_unscaled = w * (y_std / std)  # Adjust for feature scaling

# Correct Bias Scaling
b_unscaled = y_mean - np.dot(mean / std, w_unscaled)

print("\nUnscaled Weights:", w_unscaled)
print("Unscaled Bias:", b_unscaled)
# ------------------------------------------------

# --------------- Prediction Function ---------------
def predict(X_input):
    X_input_scaled = (X_input - mean) / std  # Apply same scaling as training
    scaled_prediction = f(X_input_scaled, w, b)  # Get scaled prediction
    return unscale_y(scaled_prediction, y_mean, y_std)  # Convert back

# --------------- Take User Input for Prediction ---------------
while(1):
    inp = np.array(input("Enter feature values (comma-separated): ").split(','), dtype=float)

    if(inp[0] == "end"): break

    # Ensure correct number of features
    if len(inp) != m:
        print(f"Error: Expected {m} features, but got {len(inp)}")
    else:
        print("Estimated price: ", predict(inp))
