import numpy as np
import pandas as pd
import os

# Load the cleaned data
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'labeled_lb.csv')
df = pd.read_csv(data_path)

# Extract features (first 8 columns) and labels (last 2 columns)
raw_data = df.iloc[:, :8].values
y = df[['Label_A', 'Label_B']].values

# --- 1. Data Standardization (Z-Score) ---
# This method is better than dividing by the maximum because it centers the data around zero
mean_vals = raw_data.mean(axis=0)
std_vals = raw_data.std(axis=0)
std_vals[std_vals == 0] = 1 # To prevent division by zero
x = (raw_data - mean_vals) / std_vals


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
hidden_layer_size = 64  # Increase capacity to understand complex data
output_size = 2


# --- 2. He Initialization for ReLU ---
# This equation prevents "dying" neurons at the beginning and speeds up learning
w1 = np.random.randn(input_size, hidden_layer_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_layer_size))

w2 = np.random.randn(hidden_layer_size, output_size) * np.sqrt(1. / hidden_layer_size)
b2 = np.zeros((1, output_size))


# --- 3. Training Settings ---
learning_rate = 0.1 # Very ideal rate with Z-Score
epochs = 65000

print("Training started...")

m = x.shape[0] # Number of rows (1254)

for epoch in range(epochs):
    # 1. Forward pass
    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    prediction = softmax(z2)

    # 2. Backward pass
    error = prediction - y 

    # 3. Calculate Gradients 
    dw2 = np.dot(a1.T, error) / m
    db2 = np.sum(error, axis=0, keepdims=True) / m
    
    error_hidden = np.dot(error, w2.T) 
    error_hidden[z1 <= 0] = 0  
    
    dw1 = np.dot(x.T, error_hidden) / m
    db1 = np.sum(error_hidden, axis=0, keepdims=True) / m

    # 4. Weight Update
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    
    # 5. Loss Calculation
    loss = -np.mean(np.sum(y * np.log(prediction + 1e-10), axis=1))    

    # --- Learning Rate Decay ---
    # Reduce learning rate by 10% every 2000 epochs for a soft and stable descent
    if epoch > 0 and epoch % 2000 == 0:
        learning_rate *= 0.9
        print(f"--> Learning Rate reduced to: {learning_rate:.6f}")

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4} | Loss: {loss:.6f}")

def forward_pass(x_input):
    z1 = np.dot(x_input, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    return softmax(z2)

# --- Model Accuracy Calculation ---
# 1. Pass the data one last time to get the final predictions
final_predictions = forward_pass(x)

# 2. Determine the winning server (the one with the highest probability)
predicted_servers = np.argmax(final_predictions, axis=1)

# 3. Determine the correct server from the original data
true_servers = np.argmax(y, axis=1)

# 4. Calculate the match percentage
accuracy = np.mean(predicted_servers == true_servers) * 100

print(f"\n✅ Training Complete!")
print(f"📊 Final Model Accuracy: {accuracy:.2f}%")