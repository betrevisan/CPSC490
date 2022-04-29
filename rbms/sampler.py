import torch
class SamplerRBM:
    def __init__(self):
        return
    
    def sample_hidden(self, model, visible_input):
        weighted_input = torch.mm(visible_input, model.weights.t())
        activation = weighted_input + model.hidden_bias.expand_as(weighted_input)
        prob_h_given_v = torch.sigmoid(activation)
        #return prob_h_given_v, torch.bernoulli(prob_h_given_v)
        return prob_h_given_v, self.layer_given_prob(prob_h_given_v)