import numpy as np

# Helper functions
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

class SamplerRBM:
    def __init__(self):
        return
    
    def sample_hidden(self, model, visible_input):
        weighted_input = np.dot(visible_input, model.weights)
        activation = weighted_input + model.hidden_bias
        prob_h_given_v = sigmoid(activation)
        hidden_layer = self.layer_given_prob(prob_h_given_v)
        return prob_h_given_v, hidden_layer
    
    def sample_visible(self, model, hidden_input):
        weighted_input = np.dot(hidden_input, model.weights.T)
        activation = weighted_input + model.visible_bias
        prob_v_given_h = sigmoid(activation)
        visible_layer = self.layer_given_prob(prob_v_given_h)
        return prob_v_given_h, visible_layer
    
    def layer_given_prob(self, prob):
        return np.floor(prob + np.random.rand(prob.shape[0]))
