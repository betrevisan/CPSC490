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
        """
        weighted_input = np.dot(hidden_input, model.weights.T)
        activation = weighted_input + model.visible_bias
        prob_v_given_h = sigmoid(activation)
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
        """
        weighted_input = np.dot(visible_input, model.weights)
        activation = weighted_input + model.hidden_bias
        prob_h_given_v = sigmoid(activation)
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
        """
        return np.floor(prob + np.random.rand(prob.shape[0]))
