# TibialModulationSim
A python based simulation of the bladder control circuit and the effects of tibial neuromodulation, built in [Brian2](https://github.com/brian-team/brian2).

# Network Topology
The topology of the network is based upon [previous work](https://pubmed.ncbi.nlm.nih.gov/23033877/). Neurons are simulated using a "Conductancce-Based Adaptive Exponential Linear-Integrate-and-Fire" model [(CAdEx)](https://pubmed.ncbi.nlm.nih.gov/33253029/). Synaptic connections are programmed to dynamically alter their weights according to relative spike timing.

# Bladder Simulation
The biophysical model of the bladder utilised in this simulation approach is adapted from the publication authored by [Lister et. al.](https://doi.org/10.1101/2024.11.21.624716). 

Further detail of this model, and a link to a GitHub repository may be found there. 

This model differs in that it also contains a simulated control circuit and the option to modulate the system via the tibial nerve. 

# Pipeline
The main simulation function `bladder_sim` takes the following arguments:

- time (float): The duration (in seconds) of bladder and circuit behaviour that should be produced.

- tibial_parameters (list): The tibial nerve stimulation that is to be applied to the system, formatted as [frequency(Hz), duration(s)] e.g. [1, 1000]

- parameters (dataframe): The parameters specified in the readme, dictating the weights of the synaptic connections within the model. Imported from params.py 

- pag (bool, default True): If tibial nerve projections to the Periaquaductal Grey (PAG) should be maintained (True/False)

- pmc (bool, default True): If tibial nerve projections to the Pontine Micturition Centre (PMC) should be maintained (True/False)
     
- spine_aff (bool, default True): If tibial nerve projections to spinal afferents should be maintained (True/False)
    
- random_gen (bool, default False): If the simulation should be provided a seed (i.e., returning the same results for any noise calculation). 
                                  NOTE: If this is set to true and supplied a seed the simulation will act in a non-random manner!

- rand_seed (int, default 1): Seed for any random noise calculation 

Creation of a simulation run using the `bladder_sim` function will generate the framework (i.e., neurons and synapses) for simulating the system. This will be held as the presently cached device during the simulation's run. After the simulation has been run the cache will be cleared. 


# Paramter import
A set of pre-fit synaptic weights have been supplied with this GitHub repository (as a list, in params.py). For the simulation to behave as expected these must be imported and called as an argument in the bladder_sim function. 

Details on the location of each synaptic weight within the circuit may be found within params.py. 

# Viewing Results
After compiling and running a simulation, the function will return a list of variables containing key measurements from both the bladder and control circuit. Importantly, unless otherwise specified the results are returned at a sample rate of 50Hz. In other words, each "step" of the simulation represents an increment of 20ms duration. 

The list is structured as so:


|           Variable            |                                                 Definition                                                 |
|-------------------------------|------------------------------------------------------------------------------------------------------------|
| pressure                      | Bladder pressure (cmh20) for the simulation run (array)                                                    |
| volume                        | Bladder volume (m3) for the simulation run (array)                                                         |
| bladder_pressure_mon.t/second | Timestamps associated with the bladder pressure array (for plotting as seconds without needing to convert) |
| pel_aff                       | Firing rate of pelvic afferents over the course of the simulation (hz)                                     |
| pel_out                       | Firing rate of pelvic efferents over the course of the simulation (hz)                                     |
| hyp_out                       | Firing rate of hypogastric efferents over the course of the simulation (hz)                                |
| pud_out                       | Firing rate of pudendal efferents over the course of the simulation (hz)                                   |
| asc                           | Firing rate of second order spinal afferents over the course of the simulation (hz)                        |
| asc_timestamps                | Timestamps for the second order spinal afferents to allow plotting without unit conversion                 |
| tib                           | Firing rate of the tibial nerve projections (hz)                                                           |
| pel_spikes                    | Timestamps for individual spikes from one pelvic efferent                                                  |
| hyp_spikes                    | Timestamps for individual spikes from one hypogastric efferent                                             |
| pud_spikes                    | Timestamps for individual spikes from one pudendal efferent                                                |
| des_mon                       | Firing rate for primary brainstem output (hz)                                                              |
| des_spikes                    | Timestamps for individual spikes from one primary brainstem efferent                                       |
| bladder_secondary_state_mon   | All biophysical information recorded from the bladder during the simulation                                |

  

# Dependencies
The following ypthon packages are REQUIRED to run the simulation:
- Brian2 (and ALL required dependencies)
- Numpy 
- Pandas
- SciPy
- Matplotlib
- tqdm

Additionally, to allow for standalone compilation a Cython (installed with Brian automatically), and a C++ compiler are required. Further information on setup may be found at the Brian2 wiki [HERE](https://brian2.readthedocs.io/en/latest/introduction/install.html#installation-cpp). 

# Example Operation
```python
#Import function
from Simulation import *

#Import presupplied weights
from params import parameters

#Specify desired tibial input - here 1Hz for 300s
tib_params = [1, 300]

#Call the function, saving the results to the sim variable 
sim = bladder_sim(
        time=500, #500 second runtime desired
        tibial_parameters=tib_params, #Using the tibial modulation defined above
        parameters=params, #Synaptic weights set according to the presupplied data
        pag = True, #Tibial projections to the PAG intact
        pmc = True, #Tibial projections to the PMC intact
        spine_aff = True #Tibial projections to spinal cord afferents intact
        random_gen = False, #No seed to be supplied to the system (i.e., the results will include random noise!)
        )

#At this point the simulation will run (seen as a progress report in the terminal)

#Once it has finished, we may take a look at some results, e.g., plotting urine volume over the course of the simulation
plt.plot(sim[1])
plt.xlabel("Elapsed Time (As 20ms Intervals)")
plt.ylabel("Bladder Volume (As $m^3$)")
```
![ExampleRun](https://github.com/user-attachments/assets/7f61b6b3-03d1-4567-acb7-6097ddd663c7)

# Citing This Work
If you use our repository for your published research, we kindly ask you to cite our article:

> McConnell-Trevillion Aidan, Jabbari Milad, Ju Wei, Lister Elliot, Erfanian Abbas, Mitra Srinjoy, Nazarpour Kianoush (2025) Low-Frequency Tibial Neuromodulation Increases Voiding Activity - a Human Pilot Study and Computational Model eLife 14:RP106174, https://doi.org/10.7554/eLife.106174.1

# Contact
To contact the research team, please email the corresponding author Aidan McConnell-Trevillion at:

 a.mcconnell-trevillion@ed.ac.uk

