import numpy as np

class AttentionRBM:
    """
    The class AttentionRBM class describes the restricted boltzmann machine used
    by the attention allocation model for the classic approach to the problem.
    ...
    Attributes
    ----------
    sampler : SamplerRBM
        RBM sampler
    weigths : np.Array of size (visible_dim, hidden_dim)
        Weights of the connections in the RBM
    visible_bias : np.Array of size (visible_dim)
        Biases in the visible layer of the RBM
    hidden_bias : np.Array of size (hidden_dim)
        Biases in the hidden layer of the RBM

    Methods
    -------
    train(train_data, epochs, learning_rate)
        Trains the RBM model given the number of epochs and the learning rate.
    update_weights(init_visible, init_hidden, curr_visible, curr_hidden, learning_rate)
        Update the RBM's weights given the initial visible and hidden layers, the
        current visible and hidden layers, and the learning rate.
    update_visible_bias(init_visible, curr_visible, learning_rate)
        Update the biases of the visible layer given the initial visible layer, current
        visible layer, and learning rate.
    update_hidden_bias(init_hidden, curr_hidden, learning_rate)
        Update the biases of the hidden layer given the initial hidden layer, current
        hidden layer, and learning rate.
    test(test_data, test_answers)
        Test the RBM model.
    error(answer, prediction)
        Get the error given the actual answer and the model's prediction.
    """

    def __init__(self, sampler, visible_dim, hidden_dim):
        """
        Parameters
        ----------
        sampler : SamplerRBM
            RBM sampler
        visible_dim : int
            Number of nodes in the visible dimension of the RBM
        hidden_dim : int
            Number of nodes in the hidden dimension of the RBM
        """

        self.sampler = sampler
        self.weights = np.random.rand(visible_dim, hidden_dim)
        self.visible_bias = np.random.rand(visible_dim)
        self.hidden_bias = np.random.rand(hidden_dim)
        return
    
    def train(self, train_data, epochs, learning_rate):
        """Trains the RBM model given the number of epochs and the learning rate.
        Parameters
        ----------
        train_data : np.Array
            The training data
        epochs : int
            Number of epochs to train for
        learning_rate : float
            Learning rate for training
        Returns
        -------
        None
        """
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

                # Update weights
                self.update_weights(init_visible, init_hidden, curr_visible, curr_hidden, learning_rate)

                # Update visible bias
                self.update_visible_bias(init_visible, curr_visible, learning_rate)

                # Update hidden bias
                self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)

                error += self.error(init_visible[42:], curr_visible[42:])
                n += 1.0
            
            print("epoch: " + str(epoch) + " loss: " + str(error/n))
        return
    
    def update_weights(self, init_visible, init_hidden, curr_visible, curr_hidden, learning_rate):
        """Update the RBM's weights given the initial visible and hidden layers, the
        current visible and hidden layers, and the learning rate.
        Parameters
        ----------
        init_visible : np.Array
            The initial visible layer
        init_hidden : np.Array
            The initial hidden layer
        curr_visible : np.Array
            The current visible layer
        curr_hidden : np.Array
            The current hidden layer
        learning_rate : float
            Learning rate for weight updates
        Returns
        -------
        None
        """
        positive_gradient = np.outer(init_visible, init_hidden)
        negative_gradient = np.outer(curr_visible, curr_hidden)
        self.weights += learning_rate * (positive_gradient - negative_gradient)
        return

    def update_visible_bias(self, init_visible, curr_visible, learning_rate):
        """Update the biases of the visible layer given the initial visible layer, current
        visible layer, and learning rate.
        Parameters
        ----------
        init_visible : np.Array
            The initial visible layer
        curr_visible : np.Array
            The current visible layer
        learning_rate : float
            Learning rate for bias updates
        Returns
        -------
        None
        """
        self.visible_bias += learning_rate * (init_visible - curr_visible)
        return

    def update_hidden_bias(self, init_hidden, curr_hidden, learning_rate):
        """Update the biases of the hidden layer given the initial hidden layer, current
        hidden layer, and learning rate.
        Parameters
        ----------
        init_hidden : np.Array
            The initial hidden layer
        curr_hidden : np.Array
            The current hidden layer
        learning_rate : float
            Learning rate for bias updates
        Returns
        -------
        None
        """
        self.hidden_bias += learning_rate * (init_hidden - curr_hidden)
        return

    def test(self, test_data, test_answers):
        """Test the RBM model.
        Parameters
        ----------
        test_data : np.Array
            The test data
        test_answers : np.Array
            The answers to the test data
        Returns
        -------
        None
        """
        # Testing the RBM Model
        error = 0
        n = 0.
        for i in range(len(test_data)):
            visible = test_data[i]
            visible_answer = test_answers[i]
            _,hidden = self.sampler.sample_hidden(self, visible)
            _,visible = self.sampler.sample_visible(self, hidden)
            error += self.error(visible_answer, visible[42:])
            n += 1.
        print("test loss: " + str(error/n))
        return
    
    def error(self, answer, prediction):
        """Get the error given the actual answer and the model's prediction.
        Parameters
        ----------
        answer : np.Array
            The actual correct answer
        prediction : np.Array
            The model's prediction
        Returns
        -------
        float
            The error between the actual answer and the prediction
        """
        error = np.mean(np.abs(answer - prediction))
        return error
    