# TibialModulationSim
A python based simulation of the bladder control circuit and the effects of tibial neuromodulation.

# Network Topology
The topology of the network is based upon [previous work](https://pubmed.ncbi.nlm.nih.gov/23033877/). Neurons are simulated using a "Conductancce-Based Adaptive Exponential Linear-Integrate-and-Fire" model [(CAdEx)](https://pubmed.ncbi.nlm.nih.gov/33253029/). Synaptic connections are programmed to dynamically alter their weights according to relative spike timing.

# Pipeline
The main simulation class `BladderSim` takes the following arguments:
- `bladder_dat` - A .mat (MATLAB) file containing bladder (pressure/volume), and neuronal (spike sorted) data
- `params` - A list of parameters corresponding to the initial weights of the synaptic connections (Note, a set of parameters derived from model optimisation is supplied)
- `tib_pars` - A list containing tibial modulatory parameters, in the format [frequency (Hz), duration (seconds)]
- `cores` - The number of parallel cores to use when running the simulation. Defaults to one.

Creation of an object from the `BladderSim` class will generate teh framework (i.e., neurons and synapses) for simulating the system. This will be held as the presently cached device. 

NOTE: To save this framework, it must be compiled as a standalone C++ directory using the `sim_run` method, which will also run the simulation. Initiating a second `BladderSim` object before doing this will overwrite the currently cached framework!

# Viewing Results
After compiling and running a simulation, the results may be obtained via the `results` attribute of the object. Doing so will return:
-Bladder pressure
-Pelvic afferent activity
-Hypogastric afferent activity
-Pelvic efferent activity
-Hypogastric efferent activity
-Pudendal efferent activity
-Second order sensory afferent activity
  -Including timestamps (used for optimisation)
-Activity of tibial neuromodulatory input
