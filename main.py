import numpy as np
import pandas as pd
import os

# ==========================================
# 1. Path Configuration and Model Loading
# ==========================================
# Using dynamic paths to ensure the code works seamlessly in any environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'data', 'processed', 'trained_model.npz')

try:
    model_data = np.load(model_path)
    w1 = model_data["w1"]
    b1 = model_data["b1"]
    w2 = model_data["w2"]
    b2 = model_data["b2"]
    mean_vals = model_data["mean_vals"]
    std_vals = model_data["std_vals"]
    print(" Model weights and biases loaded successfully!\n")
except FileNotFoundError:
    print(f" Error: Model file not found at {model_path}")
    print("Please make sure to run the training script (train.py) first to save the model.")
    exit()

# ==========================================
# 2. The Smart Core: Routing Function
# ==========================================
def smart_server_load_balancer(req):
    """
    Receives an array of features (metrics) and returns the optimal server
    based on the weights learned by the Neural Network.
    """
    # --- Normalize Input Data (Z-Score) ---
    x_input = (np.array(req) - mean_vals) / std_vals
    
    # --- Feed Forward Pass ---
    # Layer 1: Linear + ReLU
    z1 = np.dot(x_input, w1) + b1
    a1 = np.maximum(0, z1)
    
    # Layer 2: Linear + Softmax
    z2 = np.dot(a1, w2) + b2
    exp_z = np.exp(z2 - np.max(z2 , axis=1 , keepdims=True))
    prediction = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # Decision Making
    server_choice = np.argmax(prediction)
    return "Server A" if server_choice == 0 else "Server B"



def manual_test():
    print("\n--- Smart Server Load Balancer Test ---")
    print("Please enter the following metrics to get a routing decision:")
    try:
        # Input for the 8 required features
        traffic   = float(input("Network Traffic (MB/s): "))
        size      = float(input("Request Size (MB): "))
        threshold = float(input("Threshold: "))
        latency   = float(input("Response Time (ms): "))
        cpu_a     = float(input("CPU Load Server A (0-1): "))
        cpu_b     = float(input("CPU Load Server B (0-1): "))
        conn_a    = float(input("Connections Server A (0-1): "))
        conn_b    = float(input("Connections Server B (0-1): "))

        # Prepare the data for prediction
        new_data = np.array([[traffic, size, threshold, latency, cpu_a, cpu_b, conn_a, conn_b]])
        
        # Get prediction from your trained model
        result = smart_server_load_balancer(new_data)
        
        print("-" * 30)
        print(f"AI DECISION: Route to --> {result}")
        print("-" * 30)
        
    except ValueError:
        print("Error: Please enter numerical values only.")

if __name__ == "__main__":
    manual_test()