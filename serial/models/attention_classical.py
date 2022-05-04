import math

class AttentionModelClassical:
    """
    The class AttentionModelClassical class represents the classical model for attention alloc.
    ...
    Attributes
    ----------
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    max_dist : float
        Maximum possible distance in the coordinate plane
    Methods
    -------
    alloc_attn(dist)
        Allocates attention an attention level given a distance.
    get_attn_levels(model, agent, prey, predator)
        Gets the attention level for the agent, the prey, and the predator.
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
        self.max_dist = math.sqrt(w**2 + h**2)

    def alloc_attn(self, dist):
        """Allocates attention to a character given the distance to their target
        Parameters
        ----------
        dist : float
            The distance that will guide the allocation.
        Returns
        -------
        float
            The allocated attention level.
        Raises
        ------
        ValueError
            If no distance or a negative distance are passed.
        """

        if dist is None or dist < 0:
            raise ValueError("dist must be a non-zero number")
        
        d = dist/self.max_dist

        attention_levels = [25, 50, 75, 100]

        minimum = math.inf

        attention = 0

        # Iterate over attention levels, keeping track of the one with minimum cost
        for level in attention_levels:
            cost = -(1 - level/100)
            
            if level == 25:
                cost += -d
            elif level == 50:
                cost += -0.5*d - 0.4
            elif level == 75:
                cost += 0.5*d - 0.9
            elif level == 50:
                cost += d - 1

            if cost < minimum:
                minimum = cost
                attention = level
        
        return attention

    def get_attention_levels(self, agent, prey, predator):
        """Gets the attention level for the agent, the prey, and the predator
        Parameters
        ----------
        agent : Agent
            Agent in the predator-prey environment.
        prey : Prey
            Prey in the predator-prey environment.
        predator : Predator
            Predator in the predator-prey environment.
        Returns
        -------
        [float]
            The three allocated attention levels (agent, prey, and predator).
        Raises
        ------
        ValueError
            If characters are not passed.
        """

        if agent is None or prey is None or predator is None:
            raise ValueError("all character must be passed to the function")

        dist2prey = math.dist(agent.loc, prey.loc)
        dist2predator = math.dist(agent.loc, predator.loc)
        avg_dist = (dist2prey + dist2predator)/2

        # Agent's attention level using the average between its distance to the
        # prey and its distance to the predator.
        attn_agent = self.alloc_attn(avg_dist)

        # Prey's attention level using its distance to the agent.
        attn_prey = self.alloc_attn(dist2prey) 

        # Predator's attention level using its distance to the agent.
        attn_predator = self.alloc_attn(dist2predator)

        # Normalize attention levels so that they don't exceed 100
        total_attn = attn_agent + attn_prey + attn_predator
        attn_agent = attn_agent/total_attn * 100
        attn_prey = attn_prey/total_attn * 100
        attn_predator = attn_predator/total_attn * 100

        # Keep track of attention levels
        agent.track_attn([attn_agent, attn_prey, attn_predator])

        return [attn_agent, attn_prey, attn_predator]
