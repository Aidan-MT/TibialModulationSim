# TibialModulationSim
A python based simulation of the bladder control circuit and the effects of tibial neuromodulation, built in [Brian2](https://github.com/brian-team/brian2).

# Network Topology
The topology of the network is based upon [previous work](https://pubmed.ncbi.nlm.nih.gov/23033877/). Neurons are simulated using a "Conductancce-Based Adaptive Exponential Linear-Integrate-and-Fire" model [(CAdEx)](https://pubmed.ncbi.nlm.nih.gov/33253029/). Synaptic connections are programmed to dynamically alter their weights according to relative spike timing.

# Pipeline
The main simulation class `BladderSim` takes the following arguments:
- `bladder_dat` - A .mat (MATLAB) file containing bladder (pressure/volume), and neuronal (spike sorted) data
- `params` - A list of parameters corresponding to the initial weights of the synaptic connections (Note, a set of parameters derived from model optimisation is supplied)
- `tib_pars` - A list containing tibial modulatory parameters, in the format [frequency (Hz), duration (seconds)]
- `cores` - The number of parallel cores to use when running the simulation. Defaults to one.

Creation of an object from the `BladderSim` class will generate the framework (i.e., neurons and synapses) for simulating the system. This will be held as the presently cached device. 

NOTE: To save this framework, it must be compiled as a standalone C++ directory using the `sim_run` method, which will also run the simulation. Initiating a second `BladderSim` object before doing this will overwrite the currently cached framework!
# Data Organisation
The .mat file supplied should be structured as so:
|Data|Description|
|---|---|
|pressure|Bladder pressure (cmH2O), will be resampled to 50Hz|
|raw_signal|The spike-sorted neuronal signal (firing rate). Should be resampled to 50Hz using the `neural_preprocess` method if optimisation/fitting is planned|
|signal_time|The timestamps for the `raw_signal` info|
|tpressure|The timestamps for the `pressure` info|
|tvolume|Timestamps for the `volume` info|
|volume|Bladder volume (ml), will be resampled to 50Hz|

# Viewing Results
After compiling and running a simulation, the results may be obtained via the `results` attribute of the object. Doing so will return:
- Bladder pressure
- Pelvic afferent activity
- Hypogastric afferent activity
- Pelvic efferent activity
- Hypogastric efferent activity
- Pudendal efferent activity
- Second order sensory afferent activity
  - Including timestamps (used for optimisation)
- Activity of tibial neuromodulatory input

# Dependencies
The following ypthon packages are required to run the simulation:
- Brian2 (and all required dependencies)
- SciPy
- Pandas
- Matplotlib

Additionally, to allow for standalone compilation a Cython (installed with Brian automatically), and a C++ compiler are required. Further information on setup may be found at the Brian2 wiki [HERE](https://brian2.readthedocs.io/en/latest/introduction/install.html#installation-cpp). 

# Example Operation
```python
#Import class
from Simulation import BladderSim

#Import presupplied weights
from params import parameters

#Specify the path to the required files
dat = "path/to/bladder_data.mat"

#Specify desired tibial input - here 5Hz for 300s
tib_params = [5, 300]

#Create object, specify that 20 parallel cores will be used
sim = BladderSim(dat, parameters, tib_params, cores=20)

#At this point, the network framework has been generated - now will compile this
#Using the saveloc argument, select a directory and run
sim.sim_run(saveloc="path/to/desired/directory")

#Plot results
sim.sim_plot()
```
