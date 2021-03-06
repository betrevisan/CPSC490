"""Predator-Prey Task (Serial Approach)
This implements the Predator-Prey task within quantum computing using the 
serial approach (i.e. at each time step, allocate attention, observe locations,
and decide on optimal movement direction).
"""

from models import attention_quantum as attention_mod
from models import movement_quantum as movement_mod
from characters import agent as agent_mod
from characters import predator as predator_mod
from characters import prey as prey_mod
from metrics import metrics as metrics_mod

ITERATIONS = 10 # Number of iterations in the game
NUM_READS = 1 # Number of reads in the annealer
WIDTH = HEIGHT = 100
MAX_SPEED = 5

def main():
    # Initialize metrics instance
    metrics = metrics_mod.Metrics("Serial Quantum Implementation")

    # Initialize characters
    agent = agent_mod.Agent(WIDTH, HEIGHT)
    prey = prey_mod.Prey(WIDTH, HEIGHT)
    predator = predator_mod.Predator(WIDTH, HEIGHT)

    # Initialize the attention allocation model
    attention_model = attention_mod.AttentionModelQuantum(WIDTH, HEIGHT, NUM_READS)

    # Initialize the movement model
    movement_model = movement_mod.MovementModelQuantum(WIDTH, HEIGHT, NUM_READS)

    # Run model for n iterations
    for _ in range(ITERATIONS):
        # Allocate attention
        attn_agent, attn_prey, attn_predator = attention_model.get_attention_levels(agent,
                                                                                    prey,
                                                                                    predator)
        
        # Prey avoids agent with full attention
        prey.avoid(agent.loc, MAX_SPEED)
        # Predator pursues agent with full attention
        predator.pursue(agent.loc, MAX_SPEED)

        # Get the perceived locations
        agent_perceived = agent.perceive(agent, attn_agent)
        prey_perceived = agent.perceive(prey, attn_prey)
        predator_perceived = agent.perceive(predator, attn_predator)

        # Move the agent
        movement_model.move(agent, agent_perceived, prey_perceived, predator_perceived,
                            prey.loc, predator.loc, MAX_SPEED)

    # Add general metrics
    metrics.w = WIDTH
    metrics.h = HEIGHT
    metrics.iterations = ITERATIONS
    metrics.max_speed = MAX_SPEED
    metrics.num_reads = NUM_READS

    # Add agent metrics
    metrics.agent_alive = agent.alive
    metrics.agent_feasted = agent.feasted
    metrics.agent_loc_trace = agent.loc_trace
    metrics.agent_perceived_loc_trace = agent.perceived_agent_trace
    metrics.prey_perceived_loc_trace = agent.perceived_prey_trace
    metrics.predator_perceived_loc_trace = agent.perceived_predator_trace
    metrics.dist_agent2prey_trace = [dist[0] for dist in agent.dist_trace]
    metrics.dist_agent2predator_trace = [dist[1] for dist in agent.dist_trace]

    # Add prey metrics
    metrics.prey_alive = prey.alive
    metrics.prey_loc_trace = prey.loc_trace

    # Add predator to metrics
    metrics.predator_feasted = predator.feasted
    metrics.predator_loc_trace = predator.loc_trace

    # Add attention trace to metrics
    metrics.attention_trace = agent.attn_trace

    # Add time metrics
    metrics.total_sampling_time_attn = attention_model.sampling_time
    metrics.total_anneal_time_attn = attention_model.anneal_time
    metrics.total_readout_time_attn = attention_model.readout_time
    metrics.total_delay_time_attn = attention_model.delay_time
    metrics.total_sampling_time_move = movement_model.sampling_time
    metrics.total_anneal_time_move = movement_model.anneal_time
    metrics.total_readout_time_move = movement_model.readout_time
    metrics.total_delay_time_move = movement_model.delay_time

    # Display metrics
    print(metrics)

    return

if __name__ == "__main__":
    main()
    