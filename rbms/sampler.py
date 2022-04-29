import torch
class SamplerRBM:
    def __init__(self):
        return
    
    def sample_hidden(self, model, visible_input):
        weighted_input = torch.mm(visible_input, model.weights.t())
        activation = weighted_input + model.hidden_bias.expand_as(weighted_input)
        prob_h_given_v = torch.sigmoid(activation)
        return prob_h_given_v, self.layer_given_prob(prob_h_given_v)
    
    def sample_visible(self, model, hidden_input):
        weighted_input = torch.mm(hidden_input, model.weights)
        activation = weighted_input + model.visible_bias.expand_as(weighted_input)
        prob_v_given_h = torch.sigmoid(activation)
        return prob_v_given_h, self.layer_given_prob(prob_v_given_h)