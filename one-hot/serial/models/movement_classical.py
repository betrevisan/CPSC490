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

class MovementModelClassical:
    """
    The class MovementModelClassical class represents the classical model for movement.
    ...
    Attributes
    ----------
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    Methods
    -------
    decide_movement(agent_perceived, prey_perceived, predator_perceived, speed)
        Decide on the direction of movement given perceived locations and movement
    move(agent, agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed)
        Moves the agent into the direction decided by the classical model.
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
        self.w = w
        self.h = h
    
    def decide_movement(self, agent_perceived, prey_perceived, predator_perceived, speed):
        """Decide on the direction of movement given perceived locations and movement.
        Parameters
        ----------
        agent_perceived : [float]
            The agent's perceived location [x, y]
        prey_perceived : [float]
            The prey's perceived location [x, y]
        predator_perceived : [float]
            The predator's perceived location [x, y]
        speed : float
            The speed of movement
        Returns
        -------
        [float]
            The target position that guides the direction of movement
        """
        # Let the position of the agent be the center of the circle of radius equal to max_speed
        center = agent_perceived
        radius = speed

        # Possible directions
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        # Best reward and selected direction
        best_reward = -math.inf
        best_loc = None

        # Iterate over all possible directions keeping track of the one with the best reward
        for angle in angles:
            # Get the x and y coordinates if the agent were to move at this angle
            x = radius * np.cos(np.radians(angle)) + center[0]
            if x < 0:
                x = 0
            elif x > self.w:
                x = self.w

            y = radius * np.sin(np.radians(angle)) + center[1]
            if y < 0:
                y = 0
            elif y > self.h:
                y = self.h

            loc = [int(x), int(y)]

            # Reward = distance to predator - distance to prey
            reward = dist(loc, predator_perceived) - dist(loc, prey_perceived)
            # If this locations has the best reward seen yet, update
            if reward > best_reward:
                best_reward = reward
                best_loc = loc

        return best_loc
    
    def move(self, agent, agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed):
        """Moves the agent into the direction decided by the classical model
        Parameters
        ----------
        agent : Agent
            Agent in the predator-prey environment.
        agent_perceived : [float]
            The agent's perceived location [x, y]
        prey_perceived : [float]
            The prey's perceived location [x, y]
        predator_perceived : [float]
            The predator's perceived location [x, y]
        prey_real : [float]
            The prey's real location [x, y]
        predator_real : [float]
            The predator's real location [x, y]
        speed : float
            The speed of movement
        Returns
        -------
        void
        """
        # Get the point to move to
        move_dir = self.decide_movement(agent_perceived, prey_perceived, predator_perceived, speed)

        # Move the agent in the given direction
        agent.move(agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed, move_dir)

        return
    