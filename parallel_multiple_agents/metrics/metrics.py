class Metrics:
    """
    The Metrics class carries several metrics corresponding to an implementation
    of the predator-prey task.
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
        self.max_speed = -1
        self.visible_dim = -1
        self.hidden_dim = -1
        self.num_reads = -1
        self.training_time = 0
        self.epochs = 0
        self.learning_rate = 0
        self.training_size = 0
        self.test_size = 0
        self.train_sampling_time = 0
        self.train_anneal_time = 0
        self.train_readout_time = 0
        self.train_delay_time = 0
        self.decision_sampling_time = 0
        self.decision_anneal_time = 0
        self.decision_readout_time = 0
        self.decision_delay_time = 0
        self.decision_time = 0
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
        self.prey1_alive = True
        self.prey1_loc_trace = []
        self.prey2_alive = True
        self.prey2_loc_trace = []
        self.prey_perceived_loc_trace = []
        self.predator1_feasted = False
        self.predator1_loc_trace = []
        self.predator_perceived_loc_trace = []
        self.predator2_feasted = False
        self.predator2_loc_trace = []
        return
    
    def __repr__(self):
        """Displays information about the implementation
        """
        display = ['\n===============================']
        display.append(" ".join(self.name.upper()) + "\n")
        display.append('General Metrics')
        display.append("\tWidth x Height:                                    " + str(self.w) + " x " + str(self.h))
        display.append("\tIterations:                                        " + str(self.iterations))
        display.append("\tMax speed:                                        " + str(self.max_speed))
        display.append("\tVisible Dimensions:                                " + str(self.visible_dim))
        display.append("\tHidden Dimensions:                                " + str(self.hidden_dim))
        if self.num_reads != -1:
            display.append("\tAnnealer reads per iteration:                      " + str(self.num_reads))

        display.append('\nTraining Metrics')
        display.append('\tEpochs:                                             ' + str(self.epochs))
        display.append('\tLearning Rate:                                             ' + str(self.learning_rate))
        display.append('\tTraining Size:                                             ' + str(self.training_size))
        display.append('\tTest Size:                                             ' + str(self.test_size))

        display.append('\nTime Metrics (in microseconds)')
        if self.training_time > 0:
            display.append("\t\tTraining time:                            " + "{:.2f}".format(self.training_time))
        if self.train_sampling_time > 0:
            display.append("\t\tTraining sampling time:                            " + "{:.2f}".format(self.train_sampling_time))
            display.append("\t\tTraining anneal time:                            " + "{:.2f}".format(self.train_anneal_time))
            display.append("\t\tTraining readout time:                            " + "{:.2f}".format(self.train_readout_time))
            display.append("\t\tTraining delay time:                            " + "{:.2f}".format(self.train_delay_time))
        if self.decision_time > 0:
            display.append("\t\tAverage Decision time:                            " + "{:.2f}".format(self.decision_time/self.iterations))
        if self.decision_sampling_time > 0:
            display.append("\t\tDecision sampling time:                            " + "{:.2f}".format(self.decision_sampling_time))
            display.append("\t\tDecision anneal time:                            " + "{:.2f}".format(self.decision_anneal_time))
            display.append("\t\tDecision readout time:                            " + "{:.2f}".format(self.decision_readout_time))
            display.append("\t\tDecision delay time:                            " + "{:.2f}".format(self.decision_delay_time))


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
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
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

        display.append('\nPrey 1 Metrics')
        display.append('\tAlive:                                             ' + str(self.prey1_alive))
        display.append('\tSteps taken:                                       ' + str(len(self.prey1_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.prey1_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived prey location trace:             "
        for loc in self.prey_perceived_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('\nPrey 2 Metrics')
        display.append('\tAlive:                                             ' + str(self.prey2_alive))
        display.append('\tSteps taken:                                       ' + str(len(self.prey2_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.prey2_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived prey location trace:             "
        for loc in self.prey_perceived_loc_trace:
            loc[0] = "{:.2f}".format(loc[0])
            loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('\nPredator 1 Metrics')
        display.append('\tFeasted:                                           ' + str(self.predator1_feasted))
        display.append('\tSteps taken:                                       ' + str(len(self.predator1_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.predator1_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived predator location trace:         "
        for loc in self.predator_perceived_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('\nPredator 2 Metrics')
        display.append('\tFeasted:                                           ' + str(self.predator2_feasted))
        display.append('\tSteps taken:                                       ' + str(len(self.predator2_loc_trace) - 1))
        trace_str = "\tLocation trace:                                    "
        for loc in self.predator2_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))
        trace_str = "\tAgent's perceived predator location trace:         "
        for loc in self.predator_perceived_loc_trace:
            # loc[0] = "{:.2f}".format(loc[0])
            # loc[1] = "{:.2f}".format(loc[1])
            trace_str += ", " + str(loc)
        display.append(trace_str.replace(", ","",1))

        display.append('===============================\n')
        return "\n".join(display)
        