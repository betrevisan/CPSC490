import numpy as np
from random import randint, seed

class Prey:
    """
    The Prey class represents the prey in the Predator-Prey task.
    ...
    Attributes
    ----------
    loc : [float]
        Location of the prey [x, y]
    alive : bool
        Says whether the prey is still alive (i.e. has not been caught)
    loc_trace : [[float]]
        Keeps track of all the locations the prey was in
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    Methods
    -------
    avoid(agent_loc, speed)
        Avoids the agent given its location and a speed of movement.
    bounce_back()
        If the prey's location is outside the coordinate plane, bounce back into it.
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
        seed(13)
        self.loc = [randint(0, int(w/3)), randint(0, h)]
        self.alive = True
        self.loc_trace = [list(self.loc)]
        self.w = w
        self.h = h
        return

    def avoid(self, agent_loc, speed):
        """Avoids the agent given its location and a speed of movement.
        Parameters
        ----------
        agent_loc : [float]
            The agent's location [x, y]
        speed : float
            The speed of movement
        Returns
        -------
        void
        Raises
        ------
        ValueError
            If given arguments are invalid.
        Referenced to https://math.stackexchange.com/questions/3932112/move-a-point-along-a-vector-by-a-given-distance
        for the movement
        """
        if agent_loc is None:
            raise ValueError("agent_loc must be valid")

        if speed <= 0:
            raise ValueError("speed must be positive number")

        # If the distance between prey and predator is less than 10 it counts as a contact
        buffer = 10
        # Vector for the prey's location
        prey_v = np.array(self.loc)
        # Vector for the agent's location
        agent_v = np.array(agent_loc)
        # Vector for the direction of movement
        move_v = agent_v - prey_v
        # Distance between prey and agent
        dist2agent = np.linalg.norm(move_v)

        # If the prey has been caught, set alive to False
        if dist2agent < buffer:
            self.alive = False

        # Move prey in the opposite direction of the movement vector at the given speed
        d = speed / dist2agent
        if d > 1:
            d = 1
        new_loc = np.floor((prey_v - d * move_v))
        
        # Update prey's location
        self.loc = new_loc
        
        # If the new location is out of range, bounce back
        self.bounce_back()

        # Update prey's location trace
        self.loc_trace.append(list(self.loc))
        return
    
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
        
        return
