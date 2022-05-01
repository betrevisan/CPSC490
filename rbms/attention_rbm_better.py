import numpy as np

class AttentionRBM:
    def __init__(self, sampler, visible_dim, hidden_dim):
        self.sampler = sampler
        self.weights = np.random.rand(visible_dim, hidden_dim)
        self.visible_bias = np.random.rand(visible_dim)
        self.hidden_bias = np.random.rand(hidden_dim)
        return

    def train(self, train_data, train_data_answers, epochs, learning_rate):
        # Repeat for each epoch
        for epoch in range(epochs):
            error = 0
            n = 0.0

            # Go over each data point in the training data
            for i in range(len(train_data)):
                init_visible = train_data[i]
                init_hidden = self.sampler.sample_hidden(self, init_visible)
                curr_visible = self.sampler.sample_visible(self, init_hidden)
                curr_hidden = self.sampler.sample_hidden(self, curr_visible)
                expected_answer = train_data_answers[i]

                # Update weights
                self.update_weights(expected_answer, init_hidden, curr_visible, curr_hidden, learning_rate)

                # Update visible bias
                self.update_visible_bias(expected_answer, curr_visible, learning_rate)

                # Update hidden bias
                self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)

                error += self.error(expected_answer, curr_visible[:21])
                n += 1.0
            
            print("epoch: " + str(epoch) + " loss: " + str(error/n))
        return