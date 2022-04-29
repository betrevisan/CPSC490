import torch

class AttentionRBM:

    def __init__(self, sampler, visible_dim, hidden_dim):
        self.sampler = sampler
        self.weights = torch.randn(hidden_dim, visible_dim)
        self.hidden_bias = torch.randn(1, hidden_dim)
        self.visible_bias = torch.randn(1, visible_dim)
        self.learning_rate = 0.1
    
    def train(self, initial_visible, curr_visible, initial_prob_h, curr_prob_h):
        # CONFIRM THE WEIGHT UPDATE IS CORRECT
        self.weights += torch.mm(initial_prob_h.t(), initial_visible) - torch.mm(curr_prob_h.t(), curr_visible)
        self.visible_bias += torch.sum((initial_visible - curr_visible), 0)
        self.hidden_bias += torch.sum((initial_prob_h - curr_prob_h), 0)