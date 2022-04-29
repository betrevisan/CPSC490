# Data class for the boltzmann for attention allocation
import numpy as np

class AttentionData:
    np.random.seed(11)

    def __init__(self, train_size, test_size, width, height):
        self.train_size = train_size
        self.test_size = test_size
        self.width = width
        self.height = height
        self.train_data = self.generate_train_data()
        # self.test_data = self.generate_test_data()
        # self.train_data_bin = self.prepare_data(self.train_data)
        # self.test_data_bin = self.prepare_data(self.test_data)
        # self.train_data_bin_answers = self.prepare_train_answers()
        # self.test_data_bin_answers = self.prepare_test_answers()

    # Generate training data for the network
    def generate_train_data(self):
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, prey_attn, agent_attn, predator_attn]
        data = []
        for _ in range(self.train_size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Prey's position
            datapoint.append(self.generate_random_loc(0, self.width/3))
            # Agent's position
            datapoint.append(self.generate_random_loc(self.width/3, 2*self.width/3))
            # Predator's position
            datapoint.append(self.generate_random_loc(2*self.width/3, self.width))

            # Get the best attention allocations for the given positions
            datapoint.append(self.best_attention(datapoint))
            
            data.append(datapoint)
        
        return np.array(data)

    