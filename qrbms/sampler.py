from pyqubo import Binary
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

class SamplerQRBM:
    def __init__(self):
        return

    def sample_visible(self, model, hidden_layer):
        hamiltonian = 0
        hamiltonian_vars = []
        
        # Initialize all variables in the visible layer
        for visible_unit in range(len(model.visible_bias)):
            hamiltonian_vars.append(Binary(str(visible_unit)))
        
        for hidden_unit in range(len(model.hidden_bias)):
            # Add a connection only to the units that were activated
            if hidden_layer[hidden_unit]:
                for visible_unit in range(len(model.visible_bias)):
                    hamiltonian += -1 * model.weights[visible_unit][hidden_unit] * hamiltonian_vars[visible_unit]
              
        for visible_unit in range(len(model.visible_bias)):
            hamiltonian += -1 * model.visible_bias[visible_unit] * hamiltonian_vars[visible_unit]
        
        compiled_hamiltonian = hamiltonian.compile()
        
        bqm = compiled_hamiltonian.to_bqm()
        
        quantum_sampler = EmbeddingComposite(DWaveSampler())
        
        # CONFIRM CHAIN STRENGTH
        # CONFIRM NUM_READS
        output = quantum_sampler.sample(bqm, 2, 1)
        solution = output.first.sample
        
        # REVISIT THIS
        solution_list = [(k, v) for k, v in solution.items()]
        solution_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        visible_layer = [v for (k, v) in solution_list]
        
        return visible_layer
    
    def sample_hidden(self, model, visible_layer):
        hamiltonian = 0
        hamiltonian_vars = []
        
        # Initialize all variables in the hidden layer
        for hidden_unit in range(len(model.hidden_bias)):
            hamiltonian_vars.append(Binary(str(hidden_unit)))
        
        for visible_unit in range(len(model.visible_bias)):
            # Add a connection only to the units that were activated
            if visible_layer[visible_unit]:
                for hidden_unit in range(len(model.hidden_bias)):
                    hamiltonian += -1 * model.weights[visible_unit][hidden_unit] * hamiltonian_vars[hidden_unit]
              
        for hidden_unit in range(len(model.hidden_bias)):
            hamiltonian += -1 * model.hidden_bias[hidden_unit] * hamiltonian_vars[hidden_unit]
        
        compiled_hamiltonian = hamiltonian.compile()
        
        bqm = compiled_hamiltonian.to_bqm()
        
        quantum_sampler = EmbeddingComposite(DWaveSampler())
        
        # CONFIRM CHAIN STRENGTH
        # CONFIRM NUM_READS
        output = quantum_sampler.sample(bqm, 2, 1)
        solution = output.first.sample
        
        # REVISIT THIS
        solution_list = [(k, v) for k, v in solution.items()]
        solution_list.sort(key=lambda tup: int(tup[0]))  # sorts in place
        hidden_layer = [v for (k, v) in solution_list]
        
        return hidden_layer
