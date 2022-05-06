# Referenced to https://rubikscode.net/2018/10/22/implementing-restricted-boltzmann-machine-with-python-and-tensorflow/
# and to https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5

import numpy as np
import random

class RBM:
    """
    The class RBM class describes the restricted boltzmann machine used
    by the complete model for the classic approach to the predator-prey 
    problem.
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
        Given the visible layer (real locations), allocate attention and decide where to 
        move to based on the perceived locations.
    reconstruct(visible_layer)
        Get the reconstruction of the visible layer given its initial layer.
    get_allocations_from_binary(binary)
        Get the allocations for the prey, agent, and predator in decimal numbers given
        the binary numbers.
    get_location_from_binary(self, binary)
        Get the direction of movement in decimal numbers given the binary numbers.
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
        self.sampling_time = 0
        self.anneal_time = 0
        self.readout_time = 0
        self.delay_time = 0
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
        # Initialize sampling metrics
        sampling_time = 0
        anneal_time = 0
        readout_time = 0
        delay_time = 0

        # Repeat for each epoch
        for epoch in range(epochs):
            # Get a random datapoint
            datapoint_index = random.randrange(0, len(train_data))
            
            # Use the datapoint as the visible layer and get the hidden layer from it
            init_visible = train_data[datapoint_index]
            init_hidden, timing = self.sampler.sample_hidden(self, init_visible)
            if timing != -1:
                sampling_time += timing["qpu_sampling_time"]
                anneal_time += timing["qpu_anneal_time_per_sample"]
                readout_time += timing["qpu_readout_time_per_sample"]
                delay_time += timing["qpu_delay_time_per_sample"]

            # Get the reconstruction from the hidden layer
            curr_visible, timing = self.sampler.sample_visible(self, init_hidden)
            if timing != -1:
                sampling_time += timing["qpu_sampling_time"]
                anneal_time += timing["qpu_anneal_time_per_sample"]
                readout_time += timing["qpu_readout_time_per_sample"]
                delay_time += timing["qpu_delay_time_per_sample"]

            # Get the new hidden layer from the reconstruction
            curr_hidden, timing = self.sampler.sample_hidden(self, curr_visible)
            if timing != -1:
                sampling_time += timing["qpu_sampling_time"]
                anneal_time += timing["qpu_anneal_time_per_sample"]
                readout_time += timing["qpu_readout_time_per_sample"]
                delay_time += timing["qpu_delay_time_per_sample"]

            # Update weights
            self.update_weights(init_visible, init_hidden, curr_visible, curr_hidden, learning_rate)

            # Update visible bias
            self.update_visible_bias(init_visible, curr_visible, learning_rate)

            # Update hidden bias
            self.update_hidden_bias(init_hidden, curr_hidden, learning_rate)

            # Compute loss and display it
            error = self.error(init_visible[42:], curr_visible[42:])            
            print("epoch: " + str(epoch) + " loss: " + str(error))
        return [sampling_time, anneal_time/(3 * epochs), readout_time/(3 * epochs), delay_time/(3 * epochs)]

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
        # Compute positive and negative gradients
        positive_gradient = np.outer(init_visible, init_hidden)
        negative_gradient = np.outer(curr_visible, curr_hidden)
        # Increment weights accordingly
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
        # Increment the visible biases accordingly
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
        # Increment the hidden bias accordingly
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
            # Use the testing datapoint as the visible layer
            visible = test_data[i]
            visible_answer = test_answers[i]
            # Get the hidden layer from the input and reconstruct the input from it
            hidden, _ = self.sampler.sample_hidden(self, visible)
            visible, _ = self.sampler.sample_visible(self, hidden)

            # Compute the loss given the ideal answer and the model's
            error += self.error(visible_answer, visible[42:])
        print("test loss: " + str(error/len(test_data)))
        return
    
    def move(self, visible_layer):
        """Given the visible layer (real locations), allocate attention and decide where to 
        move to based on the perceived locations.
        Parameters
        ----------
        visible_layer : np.Array
            The visible layer
        Returns
        -------
        List
            The list of attention allocations
        List
            The location the agent should move to
        """
        # Get the reconstruction for the given input
        reconstruction = self.reconstruct(visible_layer)

        # Get the allocations from binary to decimal
        allocs_bin = reconstruction[42:63]
        allocs = self.get_allocations_from_binary(allocs_bin)

        # Get the movement location from binary to decimal
        loc_bin = reconstruction[63:]
        loc = self.get_location_from_binary(loc_bin)

        return allocs, loc

    def move_locs(self, agent, prey_loc, predator_loc, speed):
        # Initialize time metrics
        self.sampling_time = 0
        self.anneal_time = 0
        self.readout_time = 0
        self.delay_time = 0

        # Get the given locations as a binary input to the network
        locs = [prey_loc, agent.loc, predator_loc]
        input_layer = []
        for i in range(len(locs)):
            bin = []
            for j in range(2):
                if locs[i][j] >= 0:
                    # Referenced to https://www.tutorialspoint.com/binary-list-to-integer-in-python
                    bin.append([int(b) for b in list('{0:07b}'.format(int(locs[i][j])))])
            input_layer.append(np.concatenate(np.array(bin)))
        input_layer.append(np.zeros(35))

        # Get the attention and movement for the input
        allocs, move_dir = self.move(np.concatenate(input_layer))
        
        # Update attention trace
        agent.attn_trace.append(allocs)

        # Move the agent in the given direction
        agent.move(prey_loc, predator_loc, speed, move_dir)

        return [self.sampling_time, self.anneal_time, self.readout_time, self.delay_time]

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
        # Get the hidden layer from the input
        hidden_layer, timing = self.sampler.sample_hidden(self, visible_layer)
        if timing != -1:
            self.sampling_time += timing["qpu_sampling_time"]
            self.anneal_time += timing["qpu_anneal_time_per_sample"]
            self.readout_time += timing["qpu_readout_time_per_sample"]
            self.delay_time += timing["qpu_delay_time_per_sample"]

        # Reconstruct input from the hidden layer
        reconstruction, timing = self.sampler.sample_visible(self, hidden_layer)
        if timing != -1:
            self.sampling_time += timing["qpu_sampling_time"]
            self.anneal_time += timing["qpu_anneal_time_per_sample"]
            self.readout_time += timing["qpu_readout_time_per_sample"]
            self.delay_time += timing["qpu_delay_time_per_sample"]

        return reconstruction
    
    def get_allocations_from_binary(self, binary):
        """Get the allocations for the prey, agent, and predator in decimal numbers given
        the binary numbers.
        Parameters
        ----------
        binary : np.Array
            The attention allocations in binary
        Returns
        -------
        List
            The list of attention allocations in the form [prey, agent, prey]
        """
        # Referenced to https://www.tutorialspoint.com/binary-list-to-integer-in-python
        prey_binary = [int(b) for b in binary[:7]]
        prey = int("".join(str(b) for b in prey_binary), 2)
        agent_binary = [int(b) for b in binary[7:14]]
        agent = int("".join(str(b) for b in agent_binary), 2)
        predator_binary = [int(b) for b in binary[14:]]
        predator = int("".join(str(b) for b in predator_binary), 2)
        return [prey, agent, predator]
    
    def get_location_from_binary(self, binary):
        """Get the location in decimal numbers given the binary numbers.
        Parameters
        ----------
        binary : np.Array
            The location in binary
        Returns
        -------
        List
            The location in decimals
        """
        # Referenced to https://www.tutorialspoint.com/binary-list-to-integer-in-python
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
        # Mean error
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

        # Get the hidden layer from the input and reconstruct the input from it
        hidden, _ = self.sampler.sample_hidden(self, example)
        reconstruction, _ = self.sampler.sample_visible(self, hidden)

        # Compute loss
        error = self.error(example_answer, reconstruction[42:])
        print("Error: " + str(error))
        print("Ideal allocations: " + str(self.get_allocations_from_binary(example_answer[:21])))
        print("The model's allocations: " + str(self.get_allocations_from_binary(reconstruction[42:63])))
        print("Ideal movement location: " + str(self.get_location_from_binary(example_answer[21:])))
        print("The model's movement location: " + str(self.get_location_from_binary(reconstruction[63:])))
        return
    