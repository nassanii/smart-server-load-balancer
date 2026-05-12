#  Smart Server-Load Balancer AI

##  Project Overview
The Smart Server-Load Balancer is a Multi-Layer Perceptron (MLP) neural network built entirely from scratch using only standard math libraries (NumPy).

Instead of relying on high-level frameworks like TensorFlow or PyTorch, this project deconstructs the "AI Black Box". It dynamically routes incoming network requests to optimal servers (Server A vs. Server B) based on a hybrid dataset combining real-world request features and synthetic server health metrics.

## Key Engineering Highlights
- **Zero-Framework AI:** Full manual implementation of Forward Propagation and Backpropagation using the Chain Rule and Linear Algebra (Dot Products).
- **Numerical Stability:** Implemented numerically stable Softmax and Categorical Cross-Entropy Loss functions.

### Pro-Level Optimizations
- **He Initialization:** Used mathematically proven weight initialization for ReLU to prevent the "Dying ReLU" problem.
- **Z-Score Standardization:** Scaled dataset features to a zero-mean distribution to prevent Gradient zigzagging.
- **Learning Rate Decay:** Hand-coded adaptive learning rate to ensure smooth convergence and avoid overshooting the global minimum.

##  Neural Network Architecture
- **Input Layer:** 8 Features (Request Size, Network Traffic, Threshold, Response Time, CPU Load A, CPU Load B, Active Connections A, Active Connections B).
- **Hidden Layer:** 64 Neurons with ReLU Activation.
- **Output Layer:** 2 Neurons (Server A / Server B) with Softmax Activation.

##  Dataset & Results
- **Dataset:** 1,254 rows of mixed Server Logs and Telemetry data.
- **Labeling:** One-Hot Encoded targets `[1, 0]` and `[0, 1]`.
- **Performance:** Achieved a convergence loss of **0.26** and an overall routing accuracy of **~91.07%** after 65,000 epochs.

## 🔬 Mathematical Challenges Conquered (Post-Mortem)
During development, several foundational ML challenges were identified and fixed:

1. **The Double Division Bug:** Fixed gradient crushing by ensuring matrix dimensions and `1/m` averaging were applied only once at the final gradient calculation.
2. **Gradient Explosion:** Mitigated by implementing proper Batch Gradient Descent averaging rather than summed errors.
3. **Symmetry Breaking:** Replaced standard `rand()` with `randn()` to introduce negative weights, allowing the network to inhibit specific signals effectively.

##  How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nassanii/smart-server-load-balancer.git
   cd smart-server-load-balancer
   ```

2. **Ensure you have the required packages installed:**
   ```bash
   pip install numpy pandas
   ```

3. **Run the training sequence:**
   ```bash
   python main.py
   ```
   
> **Note:** Use `predict_model.py` to route a live request array through the trained weights.

---
*Built with passion by Abdurrahman Nassani as part of a deep-dive into Neural Network Mathematics and System Engineering.*