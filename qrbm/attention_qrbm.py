import numpy as np

class QRBM:
    def __init__(self, sampler, visible_dim, hidden_dim):
        self.sampler = sampler
        # Something to try: (np.random.randn(visible_dim, hidden_dim) * 2 - 1) * 1 (https://github.com/mareksubocz/QRBM/blob/master/qrbm/MSQRBM.py)
        self.weights = np.random.randn(visible_dim, hidden_dim)
        
        # Something to try: (np.random.randn(visible_dim) * 2 - 1) * 1
        self.visible_bias = np.random.randn(visible_dim)
        # Something to try: (np.random.h.randn(hidden_dim) * 2 - 1) * 1
        self.hidden_bias = np.random.randn(hidden_dim)

    def train(self, train_data, epochs, learning_rate):
        for epoch in range(epochs):
            datapoint_index = np.random.randrange(0, len(train_data))
            
            init_visible = train_data[datapoint_index]            
            init_hidden = self.sampler.sample_hidden(self, init_visible)
            
            curr_visible = self.sampler.sample_visible(self, init_hidden)
            curr_hidden = self.sampler.sample_hidden(self, curr_visible)
            
            # Should be of size (visible_dim, hidden_dim)
            positive_gradient = np.outer(init_visible, init_hidden)
            negative_gradient = np.outer(curr_visible, curr_hidden)
            
            # Update weights
            self.update_weights(positive_gradient, negative_gradient, learning_rate)
            
            # Update visible bias
            self.update_visible_bias(init_visible, curr_visible, learning_rate)
            
            # Update hidden bias
            self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)
            
            error = self.error(init_visible[0][-21:], curr_visible[0][-21:])
            print("epoch: " + str(epoch) + " error: " + str(error))
            
            return