import math
import numpy as np
from random import randint, seed

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

class Agent:
    """
    The Agent class represents the agent in the Predator-Prey task.
    ...
    Attributes
    ----------
    loc : [float]
        Location of the agent [x, y]
    feasted : bool
        Says whether the agent has caught the prey or not
    alive : bool
        Says whether the agent is still alive (i.e. has not been caught)
    loc_trace : [[float]]
        Keeps track of all the locations the agent was in
    attn_trace : [[float]]
        Keeps track of all the attention allocations
    dist_trace : [[float]]
        Keeps track of all distances at each time step
    perceived_agent_trace : [[float]]
        Keeps track of the perceived agent locations
    perceived_prey_trace : [[float]]
        Keeps track of the perceived prey locations
    perceived_predator_trace : [[float]]
        Keeps track of the perceived predator locations
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    Methods
    -------
    perceive(target, attention)
        Get the perceived location of the target given an attention level.
    move(agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed, bias)
        Move the agent using the perceived locations, speed of movement, and pursuit bias.
    bounce_back()
        If the agent's location is outside the coordinate plane, bounce back into it.
    track_attn(attention)
        Add the given set of attention levels to the attention trace.
    track_dist(attention)
        Add the given set of attention levels to the attention trace.
    """

    def __init__(self, w, h):
        """
        Parameters
        ----------
        w : int
            Width of the coordinate plane
        h : int
            Height of the coordinate plane
        """
        seed(1)
        self.loc = [50, 50]
        self.feasted = False
        self.alive = True
        self.loc_trace = [list(self.loc)]
        self.attn_trace = []
        self.dist_trace = []
        self.perceived_agent_trace = []
        self.perceived_prey_trace = []
        self.perceived_predator_trace = []
        self.w = w
        self.h = h
        return
    
    def perceive(self, target, attention):
        """Get the target's location given the attention level.
        Parameters
        ----------
        target : Agent, Prey, or Predator
            The target of attention
        attention : float
            The attention level
        Returns
        -------
        [float]
            The perceived location
        Raises
        ------
        ValueError
            If given arguments are invalid.
        """
        if target is None or attention < 0:
            raise ValueError("invalid perceived arguments")

        # Blur actual location given the allocated attention level
        blur = 100 - attention
        x = target.loc[0] + blur
        y = target.loc[1] + blur
        return [x, y]

    def move(self, prey_real, predator_real, speed, target):
        """Move the agent using the perceived locations, speed of movement, and the target.
        direction.
        Parameters
        ----------
        prey_real : [float]
            The prey's real location [x, y]
        predator_real : [float]
            The predator's real location [x, y]
        speed : float
            The speed of movement
        target : [float]
            The target position that guides the direction of movement
        Returns
        -------
        void
        Raises
        ------
        ValueError
            If given arguments are invalid.
        Referenced to https://math.stackexchange.com/questions/3932112/move-a-point-along-a-vector-by-a-given-distance
        for the movement.
        """
        if prey_real is None or predator_real is None or target is None:
            raise ValueError("locations must all be valid")

        if speed <= 0:
            raise ValueError("speed must be positive number")

        # If the distance between prey and predator is less than 10 it counts as a contact
        buffer = 10
        # Vector for the agent's real location
        agent_real_v = np.array(self.loc)
        # Vector for the prey's real location
        prey_real_v = np.array(prey_real)
        # Vector for the predator's real location
        pred_real_v = np.array(predator_real)
        # Vector for the movement's target location
        target_v = np.array(target)

        # Real distance between the agent and the predator
        real_dist2pred = np.linalg.norm(pred_real_v - agent_real_v)
        # If the agent has been caught, set alive to False
        if real_dist2pred < buffer:
            self.alive = False

        # Vector for the direction of movement
        move_v =  target_v - agent_real_v

        # Move agent alongside movement vector at a given speed
        d = speed / np.linalg.norm(move_v)
        if d > 1:
            d = 1
        new_loc = np.floor((agent_real_v + d * move_v))

        # Update agent's location
        self.loc = new_loc

        # Real distance between the prey and the agent
        real_dist2prey = np.linalg.norm(prey_real_v - np.array(self.loc))
        # If the agent has reached its prey, set feasted to True
        if real_dist2prey < buffer:
            self.feasted = True
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update location trace
        self.loc_trace.append(list(self.loc))

        # Keep track of distances
        self.track_dist([dist(prey_real, self.loc), dist(predator_real, self.loc)])

    def bounce_back(self):
        """If the location is out of range, bounces it back into range.
        Parameters
        ----------
        void
        Returns
        -------
        void
        """
        # Fix x-coordinate, if needed
        if self.loc[0] < 0:
            self.loc[0] = 1
        elif self.loc[0] > self.w:
            self.loc[0] = self.w - 1
        
        # Fix y-coordinate, if needed
        if self.loc[1] < 0:
            self.loc[1] = 1
        elif self.loc[1] > self.h:
            self.loc[1] = self.h - 1
        
    def track_attn(self, attention):
        """Add attentions to the attention_trace.
        Parameters
        ----------
        attention : [float]
            Set of attention levels to be be added to the attention trace
        Returns
        -------
        void
        """
        self.attn_trace.append(attention)

    def track_dist(self, dist):
        """ Add distances to the dist_trace.
        Parameters
        ----------
        dist : [float]
            Set of distances to be be added to the distance trace
        Returns
        -------
        void
        """
        self.dist_trace.append(dist)

    def __repr__(self):
        """Displays information about the agent.
        """
        display = ['\n===============================']
        display.append('A G E N T')
        display.append('Alive: ' + str(self.alive))
        display.append('Feasted: ' + str(self.feasted))
        display.append('Steps taken: ' + str(len(self.loc_trace)))

        display.append('Location trace:')
        loc_trace_str = ""
        for loc in self.loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            loc_trace_str += ", " + str(loc)
        display.append(loc_trace_str)

        display.append('Agent perceived location trace:')
        agent_str = ""
        for loc in self.perceived_agent_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            agent_str += ", " + str(loc)
        display.append(agent_str)

        display.append('Prey perceived location trace:')
        prey_str = ""
        for loc in self.perceived_prey_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            prey_str += ", " + str(loc)
        display.append(prey_str)

        display.append('Predator perceived location trace:')
        predator_str = ""
        for loc in self.perceived_predator_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            predator_str += ", " + str(loc)
        display.append(predator_str)

        display.append('Attention trace (agent, prey, predator):')
        attn_trace_str = ""
        for attn in self.attn_trace:
            attn[0] = "{:.2f}".format(attn[0])
            attn[1] = "{:.2f}".format(attn[1])
            attn[2] = "{:.2f}".format(attn[2])
            attn_trace_str += ", " + str(attn)
        display.append(attn_trace_str)

        display.append('Distances trace (dist to prey, dist to predator):')
        dist_trace_str = ""
        for dist in self.dist_trace:
            dist[0] = "{:.2f}".format(dist[0])
            dist[1] = "{:.2f}".format(dist[1])
            dist_trace_str += ", " + str(dist)
        display.append(dist_trace_str)

        display.append('===============================\n')
        return "\n".join(display)
    