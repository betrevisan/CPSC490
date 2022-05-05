class Metrics:

    """
    The Metrics class carries several metrics corresponding to an implementation
    of the predator-prey task.
    ...
    Attributes
    ----------
    name : str
        Name of the implementation
    Methods
    -------
    """

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            The name of the implementation
        """
        self.name = name
        self.w = 0
        self.h = 0
        self.iterations = 0
        self.num_reads = -1
        self.attention_time = 0
        self.movement_time = 0
        self.total_sampling_time_attn = 0
        self.total_anneal_time_attn = 0
        self.total_readout_time_attn = 0
        self.total_sampling_time_move = 0
        self.total_anneal_time_move = 0
        self.total_readout_time_move = 0
        self.attention_trace = []
        self.agent_alive = True
        self.agent_feasted = False
        self.agent_loc_trace = []
        self.agent_perceived_loc_trace = []
        self.dist_agent2prey_trace = []
        self.dist_agent2predator_trace = []
        self.prey_alive = True
        self.prey_loc_trace = []
        self.prey_perceived_loc_trace = []
        self.predator_feasted = False
        self.predator_loc_trace = []
        self.predator_perceived_loc_trace = []
        return
    
    def __repr__(self):
        """Displays information about the implementation
        """
        display = ['\n===============================']
        display.append(" ".join(self.name.upper()) + "\n")
        display.append('General Metrics')
        display.append("\tWidth x Height:                                    " + str(self.w) + " x " + str(self.h))
        display.append("\tIterations:                                        " + str(self.iterations))
        if self.num_reads != -1:
            display.append("\tAnnealer reads per iteration:                      " + str(self.num_reads))

        display.append('\nTime Metrics (in microseconds)')
        if self.attention_time > 0:
            display.append("\t\tAverage Attention time:                            " + "{:.2f}".format(self.attention_time/self.iterations))
            display.append("\t\tAverage Movement time:                             " + "{:.2f}".format(self.movement_time/self.iterations))
        if self.total_sampling_time_attn > 0:
            display.append("\t\tAttention sampling time:                            " + "{:.2f}".format(self.total_sampling_time_attn/self.iterations))
            display.append("\t\Attention anneal time:                             " + "{:.2f}".format(self.total_anneal_time_attn/self.iterations))
            display.append("\t\tAttention readout time:                            " + "{:.2f}".format(self.total_readout_time_attn/self.iterations))
            display.append("\t\tMovement sampling time:                             " + "{:.2f}".format(self.total_sampling_time_move/self.iterations))
            display.append("\t\tMovement anneal time:                            " + "{:.2f}".format(self.total_anneal_time_move/self.iterations))
            display.append("\t\tMovement readout time:                             " + "{:.2f}".format(self.total_readout_time_move/self.iterations))


        display.append('\nAttention Allocation Metrics')
        trace_str = "\tTrace:                                             "
        for attn in self.attention_trace:
            trace_str += ", " + str(attn)
        display.append(trace_str.replace(", ","",1))

        display.append('\nAgent Metrics')
        display.append('\tAlive:                                             ' + str(self.agent_alive))
        display.append('\tFeasted:                                           ' + str(self.agent_feasted))
        display.append('\tSteps taken:                                       ' + str(len(self.agent_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.agent_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent perceived location trace:                    "
        for loc in self.agent_perceived_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tDistance to prey trace:                            "
        for dist in self.dist_agent2prey_trace:
            trace_str += ", " + "{:.2f}".format(dist)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tDistance to predator trace:                        "
        for dist in self.dist_agent2predator_trace:
            trace_str += ", " + "{:.2f}".format(dist)
        display.append(trace_str.replace(", ","",1))

        display.append('\nPrey Metrics')
        display.append('\tAlive:                                             ' + str(self.prey_alive))
        display.append('\tSteps taken:                                       ' + str(len(self.prey_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.prey_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived prey location trace:             "
        for loc in self.prey_perceived_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('\nPredator Metrics')
        display.append('\tFeasted:                                           ' + str(self.predator_feasted))
        display.append('\tSteps taken:                                       ' + str(len(self.predator_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.predator_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived predator location trace:         "
        for loc in self.predator_perceived_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('===============================\n')
        return "\n".join(display)