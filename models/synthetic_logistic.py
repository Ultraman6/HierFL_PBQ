import torch


class LogisticRegression_SYNTHETIC(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_SYNTHETIC, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs