import math
from numpy import sqrt
from dwave.system import EmbeddingComposite, DWaveSampler

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

class AttentionModelQuantum:
    """
    The AttentionModel class represents the quantum model for attention alloc.
    ...
    Attributes
    ----------
    w : int
        Width of the coordinate plane
    h : int
        Height of the coordinate plane
    max_dist : float
        Maximum possible distance in the coordinate plane
    num_reads : int
        Number of reads in the annealer
    total_time : floar
        Total sampling time for this model
    name : str, optional
        The name of the model
    Methods
    -------
    qubo(dist)
        Updates the QUBO formulation given a distance.
    alloc_attn(dist)
        Allocates attention an attention level given a distance.
    get_attn_levels(model, agent, prey, predator)
        Gets the attention level for the agent, the prey, and the predator.
    """

    def __init__(self, w, h, num_reads, name="AttentionModel"):
        """
        Parameters
        ----------
        w : int
            Width of the coordinate plane
        h : int
            Height of the coordinate plane
        num_reads : int
            Number of reads in the annealer
        name : str, optional
            The name of the model (default is "AttentionModel")
        """

        self.w = w
        self.h = h
        self.max_dist = sqrt(w**2 + h**2)
        self.num_reads = num_reads
        self.total_time = 0
        self.name = name
    
    def qubo(self, dist):
        """Updates the QUBO given the distance to the target
        Parameters
        ----------
        dist : float
            The distance that will guide the QUBO formulation.
        Returns
        -------
        dict
            A dict with with the updated QUBO formulation.
        Raises
        ------
        ValueError
            If no distance or a negative distance are passed.
        """
        
        if dist is None or dist < 0:
            raise ValueError("dist must be a non-zero number")

        d = dist/self.max_dist

        # Attention level dependent on cost
        Q_cost = {('25','25'): -(1 - 0.25),
            ('50','50'): -(1 - 0.5),
            ('75','75'): -(1 - 0.75),
            ('100','100'): -(1 - 1),
            ('25','50'): -(-(1 - 0.25) - (1 - 0.5)),
            ('25','75'): -(-(1 - 0.25) - (1 - 0.75)),
            ('25','100'): -(-(1 - 0.25) - -(1 - 1)),
            ('50','75'): -(-(1 - 0.5) - (1 - 0.75)),
            ('50','100'): -(-(1 - 0.5) - (1 - 1)),
            ('75','100'): -(-(1 - 0.75) - (1 - 1))}

        # Attention level dependent on distance
        Q_dist = {('25','25'): -d,
            ('50','50'): -0.5*d - 0.4,
            ('75','75'): 0.5*d - 0.9,
            ('100','100'): d - 1,
            ('25','50'): -(-d -0.5*d - 0.4),
            ('25','75'): -(-d +0.5*d - 0.9),
            ('25','100'): -(-d +d - 1),
            ('50','75'): -(-0.5*d - 0.4 +0.5*d - 0.9),
            ('50','100'): -(-0.5*d - 0.4 + d - 1),
            ('75','100'): -(0.5*d - 0.9 + d - 1)}

        # Combine both QUBO formulations (cost and distance)
        Q_complete = {}
        for key in list(Q_cost.keys()):
            Q_complete[key] = Q_cost[key] + Q_dist[key]

        return Q_complete
    
    def alloc_attn(self, dist):
        """Allocates attention to a character given the distance to their target
        Parameters
        ----------
        dist : float
            The distance that will guide the QUBO formulation.
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

        # Get the QUBO formulation for the given distance
        Q = self.qubo(dist)

        # Define sampler
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Run sampler
        sampler_output = sampler.sample_qubo(Q, num_reads = self.num_reads)

        # Time statistics in microseconds
        sampling_time = sampler_output.info["timing"]["qpu_sampling_time"]
        self.total_time += sampling_time

        # Get the attention
        attn = sampler_output.record.sample[0]
        if attn[0] == 1:
            attn = 100
        elif attn[1] == 1:
            attn = 25
        elif attn[2] == 1:
            attn = 50
        else:
            attn = 75

        return attn
    
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

        dist2prey = dist(agent.loc, prey.loc)
        dist2predator = dist(agent.loc, predator.loc)
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
