# Quantum Implementation of the Predator-Prey Task in the Study of Attention

This repository contains the code base for my senior research project titled "Quantum
Implementation of the Predator-Prey Task in the Study of Attention". The research was 
conducted within the Bhattacharjee Lab at Yale University. 

The goal of the research is to implement a quantum algorithm for the Predator-Prey model
and compare its perfomance against the classical implementation of the problem considering
speed and accuracy. THe hope is that this effort will provide some insight into the
potential of quantum computing not only to cognitive algorithms but to other fields of
computer science and beyond.

The repository is divided into two approaches: serial and parallel. The serial approach is 
based on a sequential model of cognition (i.e. it considers that the agent allocates
attention and decide on movement independently from one another). On the other hand, the
parallel approach considers parallel elements of human cognition that say we are able to 
perform different processes in parallel and that the effects of one influence the other. 
As such, attention and movement are executed simultaneously.

The serial folder contains:
- classic.py: the classical implementation of the serial approach to the Predator-Prey model
- quantum.py: the quantum implementation of the serial approach to the Predator-Prey model
- characters: the folder containing a class for each of the characters in the Predator-Prey
problem (agent, prey, and predator)
- metrics: the folder containing the metrics class which displays useful information about
the algorithm (including speed, accuracy, and general structure)
- models: the folder containing the attention allocation and the movement models (both in 
classical and in quantum computing)

The parallel folder contains:
- classic.py: the classical implementation of the parallel approach to the Predator-Prey model
- quantum.py: the quantum implementation of the parallel approach to the Predator-Prey model
- characters: the folder containing a class for each of the characters in the Predator-Prey
problem (agent, prey, and predator)
- metrics: the folder containing the metrics class which displays useful information about
the algorithm (including speed, accuracy, and general structure)
- rbms: the folder containing the restricted boltzmann machine for attention allocation and
movement in parallel. It also includes sketch files of the individual attention allocation 
boltzmann machine and the inidividual movement boltzmann machine, but those are not used in
the final implementations of the parallel approach.
- samplers: the folder containing the classical sampler used in the classical restricted
boltzmann machine and the quantum sampler used in the quantum restricted boltzmann machine.
- sketches: a series of files that implement parts that led up to the final implementation
of the restricted boltzmann machine for attention and movement. 
 