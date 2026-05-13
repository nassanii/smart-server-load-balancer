import numpy as np
import pandas as pd
import os

# Load the cleaned data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'labeled_lb.csv')
df = pd.read_csv(data_path)

# Extract features (first 8 columns) and labels (last 2 columns)
raw_data = df.iloc[:, :8].values
y = df[['Label_A', 'Label_B']].values

# --- 1. Train/Test Split (70/30) & Standardization ---
np.random.seed(42) # for reproducibility in splitting
indices = np.random.permutation(raw_data.shape[0])
train_size = int(0.7 * raw_data.shape[0])
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train_raw, X_test_raw = raw_data[train_idx], raw_data[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Calculate mean and std ONLY on training data to prevent data leakage
mean_vals = X_train_raw.mean(axis=0)
std_vals = X_train_raw.std(axis=0)
std_vals[std_vals == 0] = 1 # To prevent division by zero

# Scale train and test data
X_train = (X_train_raw - mean_vals) / std_vals
X_test = (X_test_raw - mean_vals) / std_vals


# build the structure of the neural network
np.random.seed(42)  # for reproducibility

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Define the architecture
np.random.seed(42)  # for reproducibility
input_size = 8
hidden_layer_size = 16  # Increase capacity to understand complex data
output_size = 2


# --- 2. He Initialization for ReLU ---
# This equation prevents "dying" neurons at the beginning and speeds up learning
w1 = np.random.randn(input_size, hidden_layer_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_layer_size))

w2 = np.random.randn(hidden_layer_size, output_size) * np.sqrt(1. / hidden_layer_size)
b2 = np.zeros((1, output_size))


# --- 3. Training Settings ---
learning_rate = 0.1 
epochs = 200

print("Training started...")

m = X_train.shape[0] # Number of training rows

for epoch in range(epochs):
    # 1. Forward pass
    z1 = np.dot(X_train, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    prediction = softmax(z2)

    # 2. Backward pass
    error = prediction - y_train 

    # 3. Calculate Gradients 
    dw2 = np.dot(a1.T, error) / m
    db2 = np.sum(error, axis=0, keepdims=True) / m
    
    error_hidden = np.dot(error, w2.T) 
    error_hidden[z1 <= 0] = 0  
    
    dw1 = np.dot(X_train.T, error_hidden) / m
    db1 = np.sum(error_hidden, axis=0, keepdims=True) / m

    # 4. Weight Update
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    
    # 5. Loss Calculation
    loss = -np.mean(np.sum(y_train * np.log(prediction + 1e-10), axis=1))    


    if epoch % 10 == 0:
        print(f"Epoch {epoch:4} | Loss: {loss:.6f}")

def forward_pass(x_input):
    z1 = np.dot(x_input, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    return softmax(z2)

# --- Model Accuracy Calculation ---
def calculate_accuracy(x_data, y_true_data):
    predictions = forward_pass(x_data)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_true_data, axis=1)
    return np.mean(predicted_classes == true_classes) * 100

train_accuracy = calculate_accuracy(X_train, y_train)
test_accuracy = calculate_accuracy(X_test, y_test)

print(f"\n Training Complete!")
print(f" Train Accuracy: {train_accuracy:.2f}% (How well it memorized the data)")
print(f" Test Accuracy:  {test_accuracy:.2f}% (How well it generalizes to unseen data)")

#save the model inside the processed folder
np.savez(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'trained_model.npz'),
    w1=w1, b1=b1, w2=w2, b2=b2, mean_vals=mean_vals, std_vals=std_vals)

print("\n Model weights and biases saved!")