import torch

class AttentionRBM:

    def __init__(self, visible_dim, hidden_dim):
        self.weights = torch.randn(hidden_dim, visible_dim)
        self.hidden_bias = torch.randn(1, hidden_dim)
        self.visible_bias = torch.randn(1, visible_dim)
        self.learning_rate = 0.1