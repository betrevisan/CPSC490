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
    generate_random_loc()
        Generates a random position within the coordinate plane.
    best_attention(positions)
        Gets the ideal attention allocation for a given set of positions.
    get_perceived_locs(datapoint)
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

    def __init__(self, train_size, test_size, max_speed, width, height, prey_count, predator_count):
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
        self.prey_count = prey_count
        self.predator_count = predator_count
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
        # Each datapoint follows the format:
        # [agent_loc, prey_locXcount, predator_locXcount, agent_attn, prey_attnXcount, predator_attnXcount, best_loc]
        data = []
        for _ in range(self.train_size):
            datapoint = []

            # Generate random positions for the agent, prey, and predator

            # Agent's position
            datapoint.append(self.generate_random_loc())

            # Prey's position
            for _ in range(self.prey_count):
                datapoint.append(self.generate_random_loc())

            # Predator's position
            for _ in range(self.predator_count):
                datapoint.append(self.generate_random_loc())

            # Get the best attention allocations for the given positions
            datapoint += self.best_attention(datapoint[0:self.prey_count+self.predator_count+1])

            # Get the perceived locations
            perceived_locs = self.get_perceived_locs(datapoint)

            # Get the best best location to move to for the given positions
            datapoint.append(self.best_loc(perceived_locs))
            
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
        # Each datapoint follows the format:
        # [agent_loc, prey_locXcount, predator_locXcount, agent_attn, prey_attnXcount, predator_attnXcount, best_loc]
        data = []
        for _ in range(self.test_size):
            datapoint = []
            # Generate random positions for the agent, prey, and predator
            # Agent's position
            datapoint.append(self.generate_random_loc())
            # Prey's position
            for _ in range(self.prey_count):
                datapoint.append(self.generate_random_loc())
            # Predator's position
            for _ in range(self.predator_count):
                datapoint.append(self.generate_random_loc())

            # Unknown best attention allocation
            attns = [33] * (self.predator_count+self.prey_count+1)
            datapoint.append(attns)

            # Unknown best position to move to
            loc = [33, 33]
            # Padding
            loc += [-1] * (self.predator_count + self.prey_count - 1)
            datapoint.append(loc)
            
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
        loc = [np.random.randint(0, self.width), np.random.randint(0, self.height)]
        # Padding
        loc += [-1] * (self.predator_count + self.prey_count - 1)
        return loc
    
    def best_attention(self, positions):                
        """Gets the ideal attention allocation for a given set of positions.
        Parameters
        ----------
        positions : List
            List of the positions [prey, agent, predator]
        Returns
        -------
        List
            The list of attention allocations in the form [agent, prey, prey]
        """
        agent = positions[0]
        preys = positions[1:self.prey_count+1]
        predators = positions[self.prey_count+1:]

        max_dist = dist([0, 0], [self.width, self.height])
        
        # Calculate distances.
        avg_prey_distance = 0
        prey_dists = []
        for i in range(len(preys)):
            prey_dists.append(dist(preys[i], agent))
            avg_prey_distance += prey_dists[i]
            prey_dists[i] = prey_dists[i]/max_dist
        avg_prey_distance = avg_prey_distance / len(prey_dists)

        avg_predator_distance = 0
        predator_dists = []
        for i in range(len(predators)):
            predator_dists.append(dist(predators[i], agent))
            avg_predator_distance += predator_dists[i]
            predator_dists[i] = predator_dists[i]/max_dist
        avg_predator_distance = avg_predator_distance / len(predator_dists)

        ratios = [(avg_prey_distance/max_dist + avg_predator_distance/max_dist)/2]
        ratios += prey_dists
        ratios += predator_dists

        attentions_count = 1 + self.prey_count + self.predator_count
        attentions = [0] * attentions_count
        
        for i in range(attentions_count):
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
         
        # Normalize.
        total_attn = sum(attentions)

        for i in range(attentions_count):
            attentions[i] = int(attentions[i]/total_attn * 100)
     
        return [attentions]

    def get_perceived_locs(self, datapoint):
        """Gets the perceived locations based on the real locations and the allocated attention.
        Parameters
        ----------
        datapoint : List
            List of the positions of each character and the attention allocation of each
            each one of them.
        Returns
        -------
        List
            The list of perceieved locations for each character [agent, prey, predator]
        """
        locs = datapoint[:self.prey_count+self.predator_count+1]
        attentions = datapoint[self.prey_count+self.predator_count+1:]
        attentions = attentions[0]
        perceived_locs = locs

        for i in range(len(locs)):
            blur = 0.5 * (100 - attentions[i])
            perceived_locs[i][0] = locs[i][0] + blur
            perceived_locs[i][1] = locs[i][1] + blur

            if perceived_locs[i][0] > self.width:
                perceived_locs[i][0] = self.width
            
            if perceived_locs[i][1] > self.height:
                perceived_locs[i][1] = self.height
        
        return perceived_locs

    def best_loc(self, positions):
        """Gets the ideal location to move to given the set of positions.
        Parameters
        ----------
        positions : List
            List of the positions [agent, preys, predators]
        Returns
        -------
        List
            The list of coordinates of the best location to move to 
            in the form [x, y]
        """
        # Let the position of the agent be the center of the circle of radius equal to max_speed
        center = positions[0]
        radius = self.max_speed

        # All possible angles along the circumference
        angles = [x for x in range(361)]
        best_reward = -math.inf
        best_loc = None

        # Escape the closest predator and pursue the closest prey.
        preys = positions[1:self.prey_count+1]
        predators = positions[self.prey_count+1:]

        closest_prey = [-1]
        shortest_dist = None
        for i in preys:
            if closest_prey[0] == -1:
                closest_prey = i
                shortest_dist = dist(i, center)
            else:
                if dist(i, center) < shortest_dist:
                    closest_prey = i
                    shortest_dist = dist(i, center)

        closest_predator = [-1]
        shortest_dist = None
        for i in predators:
            if closest_predator[0] == -1:
                closest_predator = i
                shortest_dist = dist(i, center)
            else:
                if dist(i, center) < shortest_dist:
                    closest_predator = i
                    shortest_dist = dist(i, center)

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
            # Padding
            loc += [-1] * (self.predator_count + self.prey_count - 1)

            # Reward = distance to predator - distance to prey
            reward = dist(loc, closest_predator) - dist(loc, closest_prey)
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
        nodes_count = 2*(self.predator_count + self.prey_count + 1) + 1
        data_bin = []
        for i in range(len(target_data)):
            bin = []
            for j in range(len(target_data[i])):
                for k in range(len(target_data[i][j])):
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
            answers.append(self.train_data_bin[i][14*(self.prey_count+self.predator_count+1):])
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
            best_alloc = self.best_attention(self.test_data[i][:self.predator_count+self.prey_count+1])

            answer = list(self.test_data[i][:self.predator_count+self.prey_count+1])
            answer += best_alloc

            # Get the perceived locations
            perceived_locs = self.get_perceived_locs(answer)

            # Get the best best location to move to for the given positions
            answer.append(self.best_loc(perceived_locs))

            answer = answer[self.predator_count+self.prey_count+1:]

            answer_bin = []
            for j in range(len(answer)):
                for k in range(len(answer[j])):
                    if answer[j][k] >= 0:
                        answer_bin.append([int(b) for b in list('{0:07b}'.format(int(answer[j][k])))])
            answers.append(np.concatenate(np.array(answer_bin)))
        return np.array(answers)
