import math
import numpy as np
from dwave.system import EmbeddingComposite, DWaveSampler

class MovementModelQuantum:
    """
    The MovementModel class represents the quantum model for movement.
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
    qubo(dist2prey, dist2pred)
        Updates the QUBO formulation given distance to the prey and to the predator.
    decide_movement(agent, agent_perceived, prey_perceived, predator_perceived, speed)
        Decide on the direction of movement given perceived locations and movement.
    move(agent, agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed)
        Moves the agent into the direction decided by the quantum model.
    """

    def __init__(self, w, h, num_reads, name="MovementModel"):
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
            The name of the model (default is "MovementModel")
        """

        self.w = w
        self.h = h
        self.max_dist = np.sqrt(w**2 + h**2)
        self.num_reads = num_reads
        self.sampling_time = 0
        self.anneal_time = 0
        self.readout_time = 0
        self.delay_time = 0
        self.name = name

    def qubo(self, dist2prey, dist2predator):
        """Updates the QUBO given the distance to the target
        Parameters
        ----------
        dist2prey : [float]
            The array of distances between the prey and each of the possible directions.
        dist2predator : [float]
            The array of distances between the predator and each of the possible directions.
        Returns
        -------
        dict
            A dict with with the updated QUBO formulation.
        """

        # Build the QUBO on the prey's perceived location
        Q_prey = {}
        max_dist_prey = max(dist2prey)
        for i in range(8):
            Q_prey[(str(i),str(i))] = -(1 - dist2prey[i]/max_dist_prey)
        for i in range(8):
            for j in range(i+1, 8):
                Q_prey[(str(i),str(j))] = -(Q_prey[(str(i),str(i))] + Q_prey[(str(j),str(j))])

        # Build the QUBO on the predator's perceived location
        Q_predator = {}
        max_dist_predator = max(dist2predator)
        for i in range(8):
            Q_predator[(str(i),str(i))] = -(1 - dist2predator[i]/max_dist_predator)
        for i in range(8):
            for j in range(i+1, 8):
                Q_predator[(str(i),str(j))] = -(Q_predator[(str(i),str(i))] + Q_predator[(str(j),str(j))])

        # Combine both QUBO formulations
        Q_complete = {}
        for key in list(Q_prey.keys()):
            Q_complete[key] = Q_prey[key] + Q_predator[key]

        return Q_complete

    def decide_movement(self, agent, agent_perceived, prey_perceived, predator_perceived, speed):
        """Decide on the direction of movement given perceived locations and movement
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
        speed : float
            The speed of movement
        Returns
        -------
        [float]
            The target position that guides the direction of movement
        """
        # Calculate the possible directions of movement
        center = agent_perceived
        radius = speed
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        directions = []

        for angle in angles:
            x = radius * np.cos(np.radians(angle)) + center[0]
            y = radius * np.sin(np.radians(angle)) + center[1]
            directions.append([x, y])

        # Calculate distance between prey and directions of movement
        dist2prey = []
        for p in directions:
            dist2prey.append(math.dist(p, prey_perceived))
        
        # Calculate distance between predator and directions of movement
        dist2predator = []
        for p in directions:
            dist2predator.append(math.dist(p, predator_perceived))

        # Update QUBO formulation
        Q = self.qubo(dist2prey, dist2predator)

        # Define sampler
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Run sampler
        sampler_output = sampler.sample_qubo(Q, num_reads = self.num_reads)

        # Time statistics in microseconds
        sampling_time = sampler_output.info["timing"]["qpu_sampling_time"]
        anneal_time = sampler_output.info["timing"]["qpu_anneal_time_per_sample"]
        readout_time = sampler_output.info["timing"]["qpu_readout_time_per_sample"]
        delay_time = sampler_output.info["timing"]["qpu_delay_time_per_sample"]
        self.sampling_time += sampling_time
        self.anneal_time += anneal_time
        self.readout_time += readout_time
        self.delay_time += delay_time

        # Get the movement direction
        move_dir_idx = sampler_output.record.sample[0]
        idx = 0
        for i in range(8):
            if move_dir_idx[i] == 1:
                idx = i
                break
        move_dir = directions[idx]
        return move_dir

    def move(self, agent, agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed):
        """Moves the agent into the direction decided by the quantum model
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
        move_dir = self.decide_movement(agent, agent_perceived, prey_perceived, predator_perceived, speed)

        # Move the agent in the given direction
        agent.move(agent_perceived, prey_perceived, predator_perceived, prey_real, predator_real, speed, move_dir)

        return