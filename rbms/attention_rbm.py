import numpy as np

class AttentionRBM:

    def __init__(self, sampler, visible_dim, hidden_dim):
        self.sampler = sampler
        self.weights = np.random.rand(visible_dim, hidden_dim)
        self.visible_bias = np.random.rand(visible_dim)
        self.hidden_bias = np.random.rand(hidden_dim)
        return
    
    def train(self, train_data, epochs, learning_rate):
        for epoch in range(epochs):
            train_loss = 0
            n = 0.0
            for i in range(len(train_data)):
                init_visible = train_data[i]
                init_prob_h, init_hidden = self.sampler.sample_hidden(init_visible)
                curr_prob_v, curr_visible = self.sampler.sample_visible(init_hidden)
                curr_prob_h, curr_hidden = self.sampler.sample_hidden(curr_visible)

                self.update_weights(init_visible, init_hidden, curr_visible, curr_hidden, learning_rate)
            
                self.update_visible_bias(init_visible, curr_visible, learning_rate)
                self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)

                train_loss += np.mean(np.abs(init_visible[42:] - curr_visible[42:]))
                n += 1.0
            
            print("epoch: " + str(epoch) + " loss: " + str(train_loss/n))
        return
    
    def update_weights(self, init_visible, init_hidden, curr_visible, curr_hidden, learning_rate):
        positive_gradient = np.outer(init_visible, init_hidden)
        negative_gradient = np.outer(curr_visible, curr_hidden)
        self.weights += learning_rate * (positive_gradient - negative_gradient)
        return

    def update_visible_bias(self, init_visible, curr_visible, learning_rate):
        self.visible_bias += learning_rate * (init_visible - curr_visible)
        return

    def update_hidden_bias(self, init_hidden, curr_hidden, learning_rate):
        self.hidden_bias += learning_rate * (init_hidden - curr_hidden)
        return

    def test(self, test_data, test_answers):
        # Testing the RBM Model
        test_loss = 0
        n = 0.
        for i in range(len(test_data)):
            visible = test_data[i]
            visible_answer = test_answers[i]
            _,hidden = self.sampler.sample_hidden(visible)
            _,visible = self.sampler.sample_visible(hidden)
            test_loss += np.mean(np.abs(visible_answer - visible[42:]))
            n += 1.
        print("test loss: " + str(test_loss/n))
        return
    