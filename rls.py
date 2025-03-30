import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(num_samples=2000, alpha=0.7, L=5, seed=42):
    """
    Generate synthetic data for single-channel source separation.

    Args:
        num_samples (int): Number of samples in the time series.
        alpha (float): Mixing coefficient for the interference.
        L (int): Number of taps (window length) for RLS feature vector.
        seed (int): Random seed.

    Returns:
        x_data (Tensor): Mixture signals shaped (num_samples, L).
        s_data (Tensor): Target source signal shaped (num_samples,).
        x_raw  (Tensor): The raw mixture waveform (num_samples,).
    """
    np.random.seed(seed)
    
    # Random source and interference
    s = np.random.randn(num_samples)
    n = np.random.randn(num_samples)
    
    # Mixture
    x_raw = s + alpha * n

    # Build feature vectors: each x(t) is [x_raw[t], x_raw[t-1], ..., x_raw[t-(L-1)]]
    # We'll pad the beginning with zeros for the first L-1 samples
    x_data = []
    for t in range(num_samples):
        # Collect last L samples (or zeros if t< L)
        start = max(0, t - L + 1)
        x_window = x_raw[start:t+1]
        if len(x_window) < L:
            x_window = np.concatenate([np.zeros(L - len(x_window)), x_window])
        x_data.append(x_window)
    
    x_data = np.array(x_data)
    
    # Convert to torch Tensors
    x_data = torch.from_numpy(x_data).float()   # shape: (num_samples, L)
    s_data = torch.from_numpy(s).float()        # shape: (num_samples,)
    x_raw  = torch.from_numpy(x_raw).float()    # shape: (num_samples,)
    
    return x_data, s_data, x_raw

class RLSLayer(nn.Module):
    def __init__(self, num_taps, init_lambda=0.99):
        """
        A single iteration (layer) of RLS.

        Args:
            num_taps (int): Number of filter taps in RLS.
            init_lambda (float): Initial value of the forgetting factor.
        """
        super(RLSLayer, self).__init__()
        
        # Forgetting factor as a learnable parameter
        self.forgetting_factor = nn.Parameter(torch.tensor(init_lambda, dtype=torch.float32))
        
        # We do not explicitly store w, P in the layer because they change for each sample (or time-step).
        # They will be passed in and out during forward().
        
    def forward(self, x, d, w_prev, P_prev):
        """
        Forward pass for one RLS iteration (one time-step).

        Args:
            x (Tensor): shape (B, num_taps) or (num_taps, ) if B=1
            d (Tensor): shape (B,) desired output
            w_prev (Tensor): shape (B, num_taps) filter weights from previous step
            P_prev (Tensor): shape (B, num_taps, num_taps) inverse covariance from previous step

        Returns:
            w_next (Tensor): shape (B, num_taps) updated filter weights
            P_next (Tensor): shape (B, num_taps, num_taps) updated inverse covariance
            y_hat (Tensor): shape (B,) the estimated output at this time-step
        """
        lambda_ = torch.clamp(self.forgetting_factor, min=1e-4, max=0.9999)
        
        # Reshape if necessary for batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, num_taps)
            d = d.unsqueeze(0)  # (1, )

        # g(t) = P_prev * x / (lambda + x^T * P_prev * x)
        # For batch processing, we do a batched matrix-vector multiply
        B = x.shape[0]
        x_ = x.unsqueeze(-1)  # shape (B, num_taps, 1)
        
        Px = torch.bmm(P_prev, x_)  # shape (B, num_taps, 1)
        denom = lambda_ + torch.bmm(x_.transpose(1,2), Px).squeeze(-1).squeeze(-1)  # shape (B,)
        
        g = Px.squeeze(-1) / denom.unsqueeze(-1)  # shape (B, num_taps)
        
        # y_hat(t) = w_prev^T * x
        y_hat = torch.sum(w_prev * x, dim=-1)  # shape (B,)

        # w(t) = w(t-1) + g(t) * [ d(t) - y_hat(t) ]
        error = d - y_hat
        w_next = w_prev + g * error.unsqueeze(-1)  # shape (B, num_taps)

        # P(t) = (1/lambda) [ P(t-1) - g(t)* x^T(t) * P(t-1) ]
        # x^T(t) * P(t-1) for batch => x_.transpose(1,2) shape (B,1,num_taps)
        # so g(t) * x^T(t) is (B, num_taps, 1) * (B, 1, num_taps) => (B, num_taps, num_taps)
        gxT = g.unsqueeze(-1) * x_.transpose(1,2)
        P_update = torch.bmm(gxT, P_prev)  # shape (B, num_taps, num_taps)
        
        P_next = (1.0 / lambda_).unsqueeze(-1).unsqueeze(-1) * (P_prev - P_update)
        
        return w_next, P_next, y_hat

class DeepRLSNet(nn.Module):
    def __init__(self, num_taps, T, init_lambda=0.99):
        """
        Deep RLSNet with T unrolled layers of RLS.
        
        Args:
            num_taps (int): Filter size (# taps).
            T (int): Number of unrolled RLS steps.
            init_lambda (float): Initial forgetting factor in each layer 
                                 (you could also randomize or share across layers).
        """
        super(DeepRLSNet, self).__init__()
        self.num_taps = num_taps
        self.T = T
        
        # Create T RLS layers
        self.rls_layers = nn.ModuleList([
            RLSLayer(num_taps, init_lambda=init_lambda) for _ in range(T)
        ])
        
    def forward(self, x_seq, d_seq):
        """
        Forward pass over an entire sequence of length N using T unrolled steps.

        Args:
            x_seq (Tensor): shape (N, num_taps) or (B, N, num_taps)
            d_seq (Tensor): shape (N,) or (B, N)
            
        Returns:
            y_hats (Tensor): The RLS outputs at each step (N steps), shape (N,) or (B,N).
            w_final (Tensor): The final filter weights after T steps (or N steps if T==N).
        """
        # If we want T to be equal to the sequence length, we can set T = x_seq.shape[1] or x_seq.shape[0].
        # In a simple version, we'll assume T == N for a single sequence.
        
        # Handle batch dimension or expand
        if x_seq.dim() == 2:
            # shape (N, num_taps) => add batch dim = 1
            x_seq = x_seq.unsqueeze(0)  # (1, N, num_taps)
            d_seq = d_seq.unsqueeze(0)  # (1, N)
        
        B, N, num_taps = x_seq.shape
        
        # Initialize w, P for the batch
        w = torch.zeros(B, num_taps, dtype=x_seq.dtype, device=x_seq.device)
        P = torch.eye(num_taps, dtype=x_seq.dtype, device=x_seq.device).unsqueeze(0).repeat(B,1,1)
        
        y_hats = []
        
        # We will run for N steps if T >= N; or T steps if T < N, etc.
        # This example unrolls exactly N steps (or min(T, N)).
        steps = min(self.T, N)
        
        for t in range(steps):
            # (B, num_taps)
            x_t = x_seq[:, t, :]
            d_t = d_seq[:, t]
            
            w, P, y_hat_t = self.rls_layers[t](x_t, d_t, w, P)
            y_hats.append(y_hat_t)
        
        # If T < N, we can keep feeding the last RLS layer or do something else.
        # For simplicity, we only compute output up to 'steps'.
        
        y_hats = torch.stack(y_hats, dim=1)  # shape (B, steps)
        
        return y_hats, w


# Hyperparameters
NUM_SAMPLES = 2000
L = 5            # number of taps
ALPHA = 0.7      # mixing coefficient
T = NUM_SAMPLES  # unroll as many steps as samples
INIT_LAMBDA = 0.95
LR = 1e-2
EPOCHS = 5

# Generate synthetic data
x_data, s_data, x_raw = generate_synthetic_data(
    num_samples=NUM_SAMPLES, alpha=ALPHA, L=L, seed=42
)

# Create model
model = DeepRLSNet(num_taps=L, T=T, init_lambda=INIT_LAMBDA)

# Move to GPU if available (optional)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
x_data = x_data.to(device)
s_data = s_data.to(device)
x_raw  = x_raw.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    y_hats, w_final = model(x_data, s_data)  # y_hats shape: (1, N)
    
    # Use an average MSE across all steps as the training objective
    loss = criterion(y_hats.squeeze(0), s_data)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")


# Evaluation
model.eval()
with torch.no_grad():
    y_hats, w_final = model(x_data, s_data)
    # We'll use the final step's output
    y_hat_final = y_hats[0, -1, :]  # shape: (N,) (since the first dim is batch=1)

mse = torch.mean((y_hat_final - s_data)**2).item()
print(f"Test MSE: {mse:.6f}")

# ---------------
# PLOT & SAVE PNG
# ---------------
# Convert data to CPU for plotting if on GPU
s_data_cpu = s_data.cpu().numpy()
x_raw_cpu  = x_raw.cpu().numpy()
y_hat_final_cpu = y_hat_final.cpu().numpy()

t_axis = np.arange(NUM_SAMPLES)

plt.figure(figsize=(10, 5))
plt.plot(t_axis, s_data_cpu, label='True Source s(t)', alpha=0.7)
plt.plot(t_axis, x_raw_cpu, label='Mixture x(t)', alpha=0.5)
plt.plot(t_axis, y_hat_final_cpu, label='Estimated s(t) via DeepRLSNet', alpha=0.8)

plt.legend()
plt.title("Source Separation with DeepRLSNet")
plt.xlabel("Time Sample")
plt.ylabel("Amplitude")

# Save the figure to a PNG file
plt.savefig("deep_rls_separation_result.png", dpi=150)
plt.show()
