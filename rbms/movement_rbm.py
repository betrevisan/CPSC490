import numpy as np

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
    movem(visible_layer)
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
        self.weights = np.random.rand(visible_dim, hidden_dim)
        self.visible_bias = np.random.rand(visible_dim)
        self.hidden_bias = np.random.rand(hidden_dim)
        return