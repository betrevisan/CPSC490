import numpy as np
from pyqubo import Binary
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

class SamplerQRBM:
    """
    The class SamplerQRBM class describes the sampler used by the quantum restricted
    boltzmann machines of quantum computing.
    ...
    Attributes
    ----------
    None

    Methods
    -------
    sample_visible(model, hidden_input)
        Samples the visible layer of the model given the input from the hidden layer.
    sample_hidden(model, visible_layer)
        Samples the hidden layer of the model given the input from the visible layer.
    layer_given_sampler_output(sampler_output)
        Get the layer within the model given the output of the sampler.
    """
    def __init__(self):
        return

    def sample_visible(self, model, hidden_input):
        """Samples the visible layer of the model given the input from the hidden layer.
        Parameters
        ----------
        model : QRBM (AttentionQRBM or MovementQRBM)
            The QRBM model being sampled
        hidden_input : np.Array of size (hidden_dim)
            The input to the hidden layer
        Returns
        -------
        np.Array of size (visible_dim)
            The visible layer
        """
        hamiltonian = 0
        hamiltonian_vars = []
        
        # Initialize all variables in the visible layer
        for visible_unit in range(len(model.visible_bias)):
            hamiltonian_vars.append(Binary(str(visible_unit)))
        
        for hidden_unit in range(len(model.hidden_bias)):
            # Add a connection in the hamiltonian only to hidden units that were activated
            if hidden_input[hidden_unit]:
                # Add weights
                for visible_unit in range(len(model.visible_bias)):
                    hamiltonian += -1 * model.weights[visible_unit][hidden_unit] * hamiltonian_vars[visible_unit]
        
        # Add biases
        for visible_unit in range(len(model.visible_bias)):
            hamiltonian += -1 * model.visible_bias[visible_unit] * hamiltonian_vars[visible_unit]
        
        # Compile the hamiltonian
        compiled_hamiltonian = hamiltonian.compile()
        
        # Get BQM out of hamiltonian
        bqm = compiled_hamiltonian.to_bqm()
        
        # Initialize quantum sampler
        quantum_sampler = EmbeddingComposite(DWaveSampler())
        
        # CONFIRM CHAIN STRENGTH
        # CONFIRM NUM_READS
        # Sample BQM
        sampler_output = quantum_sampler.sample(bqm, num_reads=1, chain_strength=2)
        sampler_output = sampler_output.first.sample
        
        return self.layer_given_sampler_output(sampler_output)
    
    def sample_hidden(self, model, visible_input):
        """Samples the hidden layer of the model given the input from the visible layer.
        Parameters
        ----------
        model : QRBM (AttentionQRBM or MovementQRBM)
            The QRBM model being sampled
        hidden_input : np.Array of size (visible_dim)
            The input to the visible layer
        Returns
        -------
        np.Array of size (hidden_dim)
            The hidden layer
        """
        hamiltonian = 0
        hamiltonian_vars = []
        
        # Initialize all variables in the hidden layer
        for hidden_unit in range(len(model.hidden_bias)):
            hamiltonian_vars.append(Binary(str(hidden_unit)))
        
        for visible_unit in range(len(model.visible_bias)):
            # Add a connection in the hamiltonian only to visible units that were activated
            if visible_input[visible_unit]:
                # Add weights
                for hidden_unit in range(len(model.hidden_bias)):
                    hamiltonian += -1 * model.weights[visible_unit][hidden_unit] * hamiltonian_vars[hidden_unit]
        
        # Add biases
        for hidden_unit in range(len(model.hidden_bias)):
            hamiltonian += -1 * model.hidden_bias[hidden_unit] * hamiltonian_vars[hidden_unit]
        
        # Compile the hamiltonian
        compiled_hamiltonian = hamiltonian.compile()
        
        # Get BQM out of hamiltonian
        bqm = compiled_hamiltonian.to_bqm()
        
        # Initialize quantum sampler
        quantum_sampler = EmbeddingComposite(DWaveSampler())
        
        # CONFIRM CHAIN STRENGTH
        # CONFIRM NUM_READS
        # Sample BQM
        sampler_output = quantum_sampler.sample(bqm, num_reads=1, chain_strength=2)
        sampler_output = sampler_output.first.sample

        return self.layer_given_sampler_output(sampler_output)
    
    def layer_given_sampler_output(self, sampler_output):
        """Get the layer within the model given the output from the quantum sampler.
        Parameters
        ----------
        sampler_output : dict
            Output from the quantum sampler
        Returns
        -------
        np.Array
            The layer that the output from the quantum sampler yields
        """
        # Output as list
        sampler_output_list = [(k, v) for k, v in sampler_output.items()]

        # Sort output
        sampler_output_list.sort(key=lambda tup: int(tup[0]))

        # Get the layer
        layer = [v for (k, v) in sampler_output_list]

        return np.array(layer)
