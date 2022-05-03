import numpy as np

class Data: 
    """
    The class Data class describes the data for the complete predator-prey task.
    ...
    Attributes
    ----------
    train_size : int
        Size of the training data
    test_size : int
        Size of the test data
    max_speed : int
        Largest step the agent can take
    width : int
        Width of the coordinate plane
    height : int
        Height of the coordinate plane
    train_data : np.Array
        Training data
    train_data_bin : np.Array
        Training data in binary
    test_data_bin : np.Array
        Test data in binary
    train_data_bin_answers : np.Array
        Ideal attention allocations for the train data
    test_data_bin_answers : np.Array
        Ideal attention allocations for the test data

    Methods
    -------
    generate_train_data()
        Generates training data for the network.
    generate_test_data()
        Generates test data for the network.
    generate_random_loc(lower, upper)
        Generates a random position given the lower and the upper bounds on the x-axis.
    best_attention(positions)
        Gets the ideal attention allocation for a given set of positions.
    get_perceived_locs(positions, attention)
        Gets the perceived locations based on the real locations and the allocated attention.
    best_loc(positions)
        Gets the ideal location to move to given the set of positions.
    prepare_data_binary(target_data)
        Transforms the given data into binary.
    prepare_train_answers()
        Gets the answers to the training data.
    prepare_test_answers()
        Gets the answers to the test data.
    """
    np.random.seed(11)

    def __init__(self, train_size, test_size, max_speed, width, height):
        """
        Parameters
        ----------
        train_size : int
            Size of the training data
        test_size : int
            Size of the test data
        max_speed : int
            Largest step the agent can take
        width : int
            Width of the coordinate plane
        height : int
            Height of the coordinate plane
        """
        self.train_size = train_size
        self.test_size = test_size
        self.max_speed = max_speed
        self.width = width
        self.height = height
        # self.train_data = self.generate_train_data()
        # self.test_data = self.generate_test_data()
        # self.train_data_bin = self.prepare_data_binary(self.train_data)
        # self.test_data_bin = self.prepare_data_binary(self.test_data)
        # self.train_data_bin_answers = self.prepare_train_answers()
        # self.test_data_bin_answers = self.prepare_test_answers()
        return