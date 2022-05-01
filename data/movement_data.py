import math
import numpy as np

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

class MovementData:
    """
    The class MovementData class describes the data for the portion of 
    the problem that deals with direction of movement.
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
        self.train_data = self.generate_train_data()
        self.test_data = self.generate_test_data()
        self.train_data_bin = self.prepare_data_binary(self.train_data)
        self.test_data_bin = self.prepare_data_binary(self.test_data)
        self.train_data_bin_answers = self.prepare_train_answers()
        self.test_data_bin_answers = self.prepare_test_answers()
    
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
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, best_loc]
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

            # Get the best best location to move to for the given positions
            datapoint.append(self.best_loc(datapoint))
            
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
        # Each datapoint follows the format [prey_loc, agent_loc, predator_loc, best_loc]
        data = []
        for _ in range(self.test_size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Prey's position
            datapoint.append(self.generate_random_loc(0, self.width/3))
            # Agent's position
            datapoint.append(self.generate_random_loc(self.width/3, 2*self.width/3))
            # Predator's position
            datapoint.append(self.generate_random_loc(2*self.width/3, self.width))

            # Unknown best location
            datapoint.append([0, 0])
            
            data.append(datapoint)
        
        return np.array(data)

    def generate_random_loc(self, lower, upper):
        """Generates a random position given the lower and the upper bounds on the x-axis.
        Parameters
        ----------
        lower : float
            Lower bound on the random location's x-coordinate
        upper : float
            Upper bound on the random location's x-coordinate
        Returns
        -------
        List
            Random location as [x,y]
        """
        return [np.random.randint(lower, upper+1), np.random.randint(0, self.height)]

    def best_loc(self, positions):
        """Gets the ideal location to move to given the set of positions.
        Parameters
        ----------
        positions : List
            List of three positions [prey, agent, predator]
        Returns
        -------
        List
            The list of coordinates of the best location to move to 
            in the form [prey, agent, prey]
        """
        # Let the position of the agent be the center of the circle of radius equal to max_speed
        center = positions[1]
        radius = self.max_speed

        # All possible angles along the circumference
        angles = [x for x in range(361)]
        best_reward = -math.inf
        best_loc = None

        for angle in angles:
            # Get the x and y coordinates if the agent were to move at this angle
            x = radius * np.cos(np.radians(angle)) + center[0]
            if x < 0:
                x = 0
            elif x > self.width:
                x = self.width

            y = radius * np.sin(np.radians(angle)) + center[1]
            if y < 0:
                y = 0
            elif y > self.height:
                y = self.height

            loc = [int(x), int(y)]

            # Reward = distance to predator - distance to prey
            reward = dist(loc, positions[2]) - dist(loc, positions[0])
            # If this locations has the best reward seen yet, update
            if reward > best_reward:
                best_reward = reward
                best_loc = loc
    
        return best_loc
    
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
                for k in range(2):
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
            answers.append(self.train_data_bin[i][-14:])
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
            best_loc = self.best_loc(self.test_data[i][:3])
            best_loc_bin = []
            for j in range(len(best_loc)):
                best_loc_bin.append([int(b) for b in list('{0:07b}'.format(int(best_loc[j])))])
            answers.append(np.concatenate(np.array(best_loc_bin)))
        return np.array(answers)
    
