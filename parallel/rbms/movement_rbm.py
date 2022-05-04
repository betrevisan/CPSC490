import numpy as np
import random

class MovementRBM:
    """
    The class MovementRBM class describes the restricted boltzmann machine used
    by the direction of movement model for the classic approach to the problem.
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
    move(visible_layer)
        Decide where to move to given the visible layer.
    reconstruct(visible_layer)
        Get the reconstruction of the visible layer given its initial layer.
    get_location_from_binary(binary)
        Get the location the agent moved to in decimal numbers given
        the binary numbers.
    error(answer, prediction)
        Get the error given the actual answer and the model's prediction.
    run_example(example, example_answer)
        Runs one specific example throught the model.
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
        self.weights = np.random.randn(visible_dim, hidden_dim)
        self.visible_bias = np.random.randn(visible_dim)
        self.hidden_bias = np.random.randn(hidden_dim)
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
            datapoint_index = random.randrange(0, len(train_data))
            
            init_visible = train_data[datapoint_index]
            init_hidden = self.sampler.sample_hidden(self, init_visible)

            curr_visible = self.sampler.sample_visible(self, init_hidden)
            curr_hidden = self.sampler.sample_hidden(self, curr_visible)

            # Update weights
            self.update_weights(init_visible, init_hidden, curr_visible, curr_hidden, learning_rate)

            # Update visible bias
            self.update_visible_bias(init_visible, curr_visible, learning_rate)

            # Update hidden bias
            self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)

            error = self.error(init_visible[42:], curr_visible[42:])            
            print("epoch: " + str(epoch) + " loss: " + str(error))
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
        for i in range(len(test_data)):
            visible = test_data[i]
            visible_answer = test_answers[i]
            hidden = self.sampler.sample_hidden(self, visible)
            visible = self.sampler.sample_visible(self, hidden)
            error += self.error(visible_answer, visible[42:])
        print("test loss: " + str(error/len(test_data)))
        return
    
    def move(self, visible_layer):
        """Decide where to move to given the visible layer.
        Parameters
        ----------
        visible_layer : np.Array
            The visible layer
        Returns
        -------
        List
            The location the agent should move to as [x, y]
        """
        reconstruction = self.reconstruct(visible_layer)
        location_bin = reconstruction[42:]
        allocations = self.get_location_from_binary(location_bin)
        return allocations
    
    def reconstruct(self, visible_layer):
        """Get the reconstruction of the visible layer given its initial layer.
        Parameters
        ----------
        visible_layer : np.Array
            The visible layer
        Returns
        -------
        np.Array
            The reconstructed visible layer
        """
        hidden_layer = self.sampler.sample_hidden(self, visible_layer)
        reconstruction = self.sampler.sample_visible(self, hidden_layer)
        return reconstruction
    
    def get_location_from_binary(self, binary):
        """Get the location the agent moved to in decimal numbers given
        the binary numbers.
        Parameters
        ----------
        binary : np.Array
            The location in binary
        Returns
        -------
        List
            The location the agent should move to as [x, y]
        """
        x_binary = [int(b) for b in binary[:7]]
        x = int("".join(str(b) for b in x_binary), 2)
        y_binary = [int(b) for b in binary[7:]]
        y = int("".join(str(b) for b in y_binary), 2)
        return [x, y]

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

    def run_example(self, example, example_answer):
        """Runs one specific example throught the model.
        Parameters
        ----------
        example : np.Array
            The example
        example_answer : np.Array
            The correct answer to the example
        Returns
        -------
        None
        """
        error = 0
        hidden = self.sampler.sample_hidden(self, example)
        reconstruction = self.sampler.sample_visible(self, hidden)
        error = self.error(example_answer, reconstruction[42:])
        print("Error: " + str(error))
        print("Ideal movement location: " + str(self.get_location_from_binary(example_answer)))
        print("The model's movement location: " + str(self.get_location_from_binary(reconstruction[42:])))
        return