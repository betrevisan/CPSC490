# Referenced to https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
# and to https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
# for the framework of this implementation

import numpy as np

# Helper functions
def sigmoid(x):
    """Computes the sigmoid function.
    Parameters
    ----------
    x : float
        Sigmoid input.
    Returns
    -------
    float
        Sigmoid of the input
    """
    y = 1 / (1 + np.exp(-x))
    return y

class SamplerRBM:
    """
    The class SamplerRBM class describes the sampler used by the restricted boltzmann
    machines of classic computing.
    ...
    Attributes
    ----------
    None

    Methods
    -------
    sample_visible(model, hidden_input)
        Samples the visible layer of the model given the input from the hidden layer.
    sample_hidden(model, visible_input)
        Samples the hidden layer of the model given the input from the visible layer.
    layer_given_prob(prob)
        Get the layer within the model given its probability distribution.
    """

    np.random.seed(11)

    def __init__(self):
        return

    def sample_visible(self, model, hidden_input):
        """Samples the visible layer of the model given the input from the hidden layer.
        Parameters
        ----------
        model : RBM (AttentionRBM or MovementRBM)
            The RBM model being sampled
        hidden_input : np.Array of size (hidden_dim)
            The input to the hidden layer
        Returns
        -------
        np.Array of size (visible_dim)
            The visible layer
        int
            -1 because the quantum implementation returns time information on the second 
            return value but timining is done in a different way in the classical algorithm

        Referenced to https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
        and to https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
        """
        # Get the weighted input given the hidden layer and the weights
        weighted_input = np.dot(hidden_input, model.weights.T)
        # Add the biases from the visible layer to form the activation input
        activation = weighted_input + model.visible_bias
        # Pass the activation input into the activation function to get the probability distribution
        prob_v_given_h = sigmoid(activation)
        # Get the visible layer from the probability distribution
        visible_layer = self.layer_given_prob(prob_v_given_h)
        return visible_layer, -1
    
    def sample_hidden(self, model, visible_input):
        """Samples the hidden layer of the model given the input from the visible layer.
        Parameters
        ----------
        model : RBM (AttentionRBM or MovementRBM)
            The RBM model being sampled
        visible_input : np.Array of size (visible_dim)
            The input to the visible layer
        Returns
        -------
        np.Array of size (hidden_dim)
            The hidden layer
        int
            -1 because the quantum implementation returns time information on the second 
            return value but timining is done in a different way in the classical algorithm
        
        Referenced to https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
        and to https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
        """
        # Get the weighted input given the visible layer and the weights
        weighted_input = np.dot(visible_input, model.weights)
        # Add the biases from the hidden layer to form the activation input
        activation = weighted_input + model.hidden_bias
        # Pass the activation input into the activation function to get the probability distribution
        prob_h_given_v = sigmoid(activation)
        # Get the hidden layer from the probability distribution
        hidden_layer = self.layer_given_prob(prob_h_given_v)
        return hidden_layer, -1
    
    def layer_given_prob(self, prob):
        """Get the layer within the model given its probability distribution.
        Parameters
        ----------
        prob : np.Array
            The probability distribution
        Returns
        -------
        np.Array
            The layer that the probability distribution yields
        
        Referenced to https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
        and to https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5
        """
        return np.floor(prob + np.random.rand(prob.shape[0]))
