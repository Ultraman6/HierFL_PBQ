import torch

class LogisticRegression_MNIST(torch.nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super(LogisticRegression_MNIST, self).__init__()
        # No need to multiply input_dim by 28*28 anymore
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Adjust the view only if input_dim is 784 (for MNIST)
        if x.shape[1] != 784:
            outputs = self.linear(x)
        else:
            x = x.view(-1, 784)  # Only reshape for MNIST
            outputs = self.linear(x)
        return outputs
