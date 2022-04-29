import torch

class AttentionRBM:

    def __init__(self, sampler, visible_dim, hidden_dim):
        self.sampler = sampler
        self.weights = torch.randn(hidden_dim, visible_dim)
        self.visible_bias = torch.randn(1, visible_dim)
        self.hidden_bias = torch.randn(1, hidden_dim)
        self.learning_rate = 0.1
    
    def train(self, train_data, epochs, initial_visible, curr_visible, initial_prob_h, curr_prob_h):
        train_data = torch.FloatTensor(train_data)

        for epoch in range(0, epochs):
            train_loss = 0
            n = 0.0
            for i in range(len(train_data)):
                init_visible = train_data[i:i+1]
                init_prob_h, init_hidden = self.sampler.sample_hidden(init_visible)
                curr_prob_v, curr_visible = self.sampler.sample_visible(init_hidden)
                curr_prob_h, curr_hidden = self.sampler.sample_hidden(curr_visible)
                
                # CONFIRM THE WEIGHT UPDATE IS CORRECT
                self.weights += torch.mm(init_prob_h.t(), init_visible) - torch.mm(curr_prob_h.t(), curr_visible)
                self.visible_bias += torch.sum((init_visible - curr_visible), 0)
                self.hidden_bias += torch.sum((init_prob_h - curr_prob_h), 0)

                train_loss += torch.mean(torch.abs(init_visible[0][42:] - curr_visible[0][42:]))
                n += 1.0
            
            print("epoch: " + str(epoch) + " loss: " + str(train_loss/n))
