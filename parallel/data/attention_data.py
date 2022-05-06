import numpy as np
import math

# Helper functions
def dist(p1, p2):
    """Computes the distance between two points.
    Parameters
    ----------
    p1 : List
        Coordinates of point #1 as [x, y]
    p2 : List
        Coordinates of point #2 as [x, y]
    Returns
    -------
    float
        The distance between p1 and p2
    """
    return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))

class AttentionData:
    """
    The class AttentionData class describes the data for the portion of 
    the problem that deals with attention allocation.
    ...
    Attributes
    ----------
    train_size : int
        Size of the training data
    test_size : int
        Size of the test data
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
    generate_random_loc()
        Generates a random position within the coordinate plane.
    best_attention(positions)
        Gets the ideal attention allocation for a given set of positions.
    prepare_data_binary(target_data)
        Transforms the given data into binary.
    prepare_train_answers()
        Gets the answers to the training data.
    prepare_test_answers()
        Gets the answers to the test data.
    """
    np.random.seed(11)

    def __init__(self, train_size, test_size, width, height):
        """
        Parameters
        ----------
        train_size : int
            Size of the training data
        test_size : int
            Size of the test data
        width : int
            Width of the coordinate plane
        height : int
            Height of the coordinate plane
        """
        self.train_size = train_size
        self.test_size = test_size
        self.width = width
        self.height = height
        self.train_data = self.generate_train_data()
        self.test_data = self.generate_test_data()
        self.train_data_bin = self.prepare_data_binary(self.train_data)
        self.test_data_bin = self.prepare_data_binary(self.test_data)
        self.train_data_bin_answers = self.prepare_train_answers()
        self.test_data_bin_answers = self.prepare_test_answers()
        return

    def generate_train_data(self):
        """Generates training data for the network.
        Parameters
        ----------
        None
        Returns
        -------
        np.Array
            Training data
        """
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, prey_attn,
        # agent_attn, predator_attn]
        data = []
        for _ in range(self.train_size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Prey's position
            datapoint.append(self.generate_random_loc())
            # Agent's position
            datapoint.append(self.generate_random_loc())
            # Predator's position
            datapoint.append(self.generate_random_loc())

            # Get the best attention allocations for the given positions
            datapoint.append(self.best_attention(datapoint))
            
            data.append(datapoint)
        
        return np.array(data)
    
    def generate_test_data(self):
        """Generates test data for the network.
        Parameters
        ----------
        None
        Returns
        -------
        np.Array
            Test data
        """
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, prey_attn,
        # agent_attn, predator_attn]
        data = []
        for _ in range(self.test_size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Prey's position
            datapoint.append(self.generate_random_loc())
            # Agent's position
            datapoint.append(self.generate_random_loc())
            # Predator's position
            datapoint.append(self.generate_random_loc())

            # Unknown best attention allocation
            datapoint.append([33, 33, 33])
            
            data.append(datapoint)
        
        return np.array(data)
    
    def generate_random_loc(self):
        """Generates a random position within the coordinate plane.
        Parameters
        ----------
        Noen
        Returns
        -------
        List
            Random location as [x,y]
        """
        return [np.random.randint(0, self.width), np.random.randint(0, self.height), -1]
    
    def best_attention(self, positions):                
        """Gets the ideal attention allocation for a given set of positions.
        Parameters
        ----------
        positions : List
            List of three positions [prey, agent, predator]
        Returns
        -------
        List
            The list of attention allocations in the form [prey, agent, prey]
        """
        dist2prey = dist(positions[0], positions[1])
        dist2predator = dist(positions[1], positions[2])
        max_dist = dist([0, 0], [self.width, self.height])
        ratios = [dist2prey/max_dist, (dist2prey/max_dist + dist2predator/max_dist)/2, dist2predator/max_dist]
        attentions = [0, 0, 0]
        
        for i in range(3):
            best_attention = 0
            best_cost = math.inf
            
            for attention in [25, 50, 75, 100]:
                cost = -(1 - attention/100)
                
                if attention == 25:
                    cost += -ratios[i]
                elif attention == 50:
                    cost += -0.5*ratios[i] - 0.4
                elif attention == 75:
                    cost += 0.5*ratios[i] - 0.9
                elif attention == 50:
                    cost += ratios[i] - 1
                    
                if cost < best_cost:
                    best_cost = cost
                    best_attention = attention
            
            attentions[i] = best_attention
         
        total_attn = attentions[0] + attentions[1] + attentions[2]
        attentions[0] = int(attentions[0]/total_attn * 100)
        attentions[1] = int(attentions[1]/total_attn * 100)
        attentions[2] = int(attentions[2]/total_attn * 100)
     
        return attentions
    
    def prepare_data_binary(self, target_data):
        """Transforms the given data into binary.
        Parameters
        ----------
        target_data : np.Array
            Data in decimal
        Returns
        -------
        np.Array
            Data in binary
        """
        data_bin = []
        for i in range(len(target_data)):
            bin = []
            for j in range(4):
                for k in range(3):
                    if target_data[i][j][k] >= 0:
                        bin.append([int(b) for b in list('{0:07b}'.format(int(target_data[i][j][k])))])
            data_bin.append(np.concatenate(np.array(bin)))
        return np.array(data_bin)
    
    def prepare_train_answers(self):
        """Gets the answers to the training data.
        Parameters
        ----------
        None
        Returns
        -------
        np.Array
            Answers to the training data
        """
        answers = []
        for i in range(len(self.train_data_bin)):
            answers.append(self.train_data_bin[i][42:])
        return np.array(answers)

    def prepare_test_answers(self):
        """Gets the answers to the test data.
        Parameters
        ----------
        None
        Returns
        -------
        np.Array
            Answers to the test data
        """
        answers = []
        for i in range(len(self.test_data_bin)):
            best_alloc = self.best_attention(self.test_data[i][:3])
            best_alloc_bin = []
            for j in range(len(best_alloc)):
                best_alloc_bin.append([int(b) for b in list('{0:07b}'.format(int(best_alloc[j])))])
            answers.append(np.concatenate(np.array(best_alloc_bin)))
        return np.array(answers)
