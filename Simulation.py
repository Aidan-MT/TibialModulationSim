"""
This file contains all the required functions necessary for the running of a simulation. 

The parameters file, as specified in the readme contains A list of parameters corresponding to the initial weights of the synaptic 
connections (including lower and upper neuroplastic bounds)

Biophysical bladder model adapted from Lister et al. 2024 (DOI: 10.1101/2024.11.21.624716)
"""

# First import required packages
# Base
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm
from scipy import optimize
import sys

# External data (Input/Output)
from scipy import io

# Data Smoothing
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d

# Plotting
from matplotlib.lines import Line2D

# Computational efficiency
prefs.codegen.target = 'numpy' 

# Define a class to handle simulation progress reporting
class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")


# Define Biophysical Bladder Model 
class LUT:
    def __init__(self):
        self.V_B = 0
        self.f_aD_s = 0
        self.f_aS_s = 0
        self.r_U = 0
        self.Q = 0
        self.Q_in = 0
        self.p_D = 0
        self.p_S = 0
        self.voiding = False
        self.w_s_s = 0
        self.w_i_s = 0
        self.w_e_s = 0
        self.u_D = 0
        self.bladder_args = {
        # Constants
        'alpha' : 2,
        'A_BN' : 7.07 * 10 ** -4,
        'A_C' : 8.0 * 10 ** -5,
        'A_muscleS' : 8.0 * 10 ** -6,
        'A_nomD' : 2.78 * 10 ** -4,
        'A_tissueS' : 4.0 * 10 ** -5,
        'C_l' : 5.0 * 10 ** 3,
        'C_u' : 2.0 * 10 ** 2,
        'C_Qin' : 5.0 * 10 ** -8,
        'C_p' : 1.5,
        'dr' : 1.0 * 10 ** -4,
        'h_D' : 8.19 * 10 ** -4,
        'h_nomS' : 2.65 * 10 ** -4,
        'k' : 0.3,
        'l_optD' : 4.68 * 10 ** -6,
        'l_optS' : 2.23 * 10 ** -6,
        'p_0S' : 100,
        'p_nomS' : 6.0 * 10 ** 3,
        'p_theta' : 3.0 * 10 ** 3,
        'rho' : 1.0 * 10 ** 3,
        'r_BN' : 1.5 * 10 ** -2,
        'r_optD' : 5.4 * 10 ** -2,
        'r_optS' : 4.8 * 10 ** -3,
        'r_0D' : 2.7 * 10 ** -2,
        'r_0S' : 4.8 * 10 ** -3,
        'R_1' : 3.0 * 10 ** 8,
        'R_2' : 2.4 * 10 ** 8,
        'sigma_isoD' : 4.0 * 10 ** 5,
        'sigma_isoS' : 2.0 * 10 ** 5,
        'tau_D' : 1.0,
        'tau_S' : 0.2,
        'u_maxD' : 0.2,
        'u_maxS' : 1.0,
        'V_muscleD' : 3.0 * 10 ** -5,
        'V_tissueD' : 2.0 * 10 ** -5,
        'tissue_pressed_in_BN' : False
        }

        self.bladder_args['max_V_B'] = 5 * 10 ** -4
        self.bladder_args['voiding_threshold'] = 1 * self.bladder_args['max_V_B']
        self.bladder_args['neuron_threshold'] = 0.50*self.bladder_args['voiding_threshold']
        self.bladder_args['filling_phase_I'] = 0.04 * self.bladder_args['max_V_B']
        self.bladder_args['filling_phase_II'] = 0.75 * self.bladder_args['max_V_B']
        self.bladder_args['filling_phase_III'] = 0.9 * self.bladder_args['max_V_B']

        self.n = 1
        self.t = 0

        self.inputvals = [] #Placeholder, external values I'm going to play with

    def get_p_S(self, A_U, f_aS_s, r_U):
        r_outS = ((1 / np.pi) * (A_U + self.bladder_args['A_tissueS'] + self.bladder_args['A_muscleS']) ) ** (1/2) # Eq. (B.10)
        r_inS = ((1 / np.pi) * (A_U + self.bladder_args['A_tissueS']) ) ** (1/2) # Eq. (B.11)
        r_S = (r_outS + r_inS) / 2 # Eq. (B.9)
        h_S = r_outS - r_inS # Mentioned below Eq. (7)

        dru = (r_U - self.r_U) / self.bladder_args['dT'] # Eq. (B.49)
        u_S = - r_U / (2 * self.bladder_args['r_optS']) * (1 / r_outS + 1 / r_inS) * dru # Eq. (B.50)
        u_S_s = u_S / self.bladder_args['u_maxS']

        l_S = (self.bladder_args['l_optS'] / self.bladder_args['r_optS']) * r_S # Eq. (B.12)    
        sigma_nom_actS = f_aS_s * self.bladder_args['sigma_isoS'] * self.sigma_u_s(u_S_s) * self.sigma_lS_s(l_S) # Eq. (7) removed passive component
        sigma_actS = (self.bladder_args['h_nomS'] / h_S) * sigma_nom_actS # Eq. (8) removed passive component

        p_actS = sigma_actS * np.log(r_outS / r_inS) # Eq. (11)
        p_pasS = self.bladder_args['p_0S'] * (self.bladder_args['p_nomS'] / self.bladder_args['p_0S']) ** (r_U / self.bladder_args['r_0S']) # Eq. (B.33)
        p_S = p_actS + p_pasS # Eq. (12)
        return p_S

    def get_Q2(self, A_U, A_T, p_S):
        RA2 = self.bladder_args['R_1'] * A_U + (self.bladder_args['R_2'] / self.bladder_args['A_C']) * (A_U ** 2) # Eq. (B.48)
        Q2 = (p_S * A_U ** 2) / ((self.bladder_args['rho'] / 2) * (1 - (A_U ** 2 / A_T ** 2)) + RA2) # Eq. (B.52)
        return Q2

    def get_Qin(self, t):
        """
        This function generates a stochastic inflow to the bladder.
        """
        # Stochastic inflow
        ## Parameters
        a = 0.025
        b = 2 * np.pi / 24
        c = 0.05
        p = 1/(60*60)
        o = 8 * 2 * np.pi / 24 * 1/p
        Q = a * np.sin(p * (b * t - o)) + c
        # Noise
        ## Noise is randomised every minute of operation to prevent overly smooth noise when dt is small
        if t % 60 == 0:
            self.n = np.random.uniform(-1, 1)
        Q = (Q * 10 ** -6) # Scale inflow from ml to m^3
        Q += self.n * self.bladder_args['C_Qin']
        return Q

    def f_0(self, V_B, f_aD_s, f_aS_s, r_U):
        A_U = np.pi * r_U ** 2 # Eq. (B.8)
        A_T = np.pi * (r_U + self.bladder_args['dr']) ** 2 # Eq. (B.15)
        A_BN = np.pi * self.bladder_args['r_BN'] ** 2 # Eq. (B.14)
        
        r_inD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD']) ) ** (1/3) # Eq. (B.4)
        A_inD = 4 * np.pi * r_inD ** 2 # Eq. (B.39)

        p_S = self.get_p_S(A_U, f_aS_s, r_U)

        Q2 = self.get_Q2(A_U, A_T, p_S)
        p_D = self.get_p_D(V_B, f_aD_s, np.sqrt(Q2))

        p_BN = p_D - ((self.bladder_args['rho'] * Q2) / 2) * (1 / A_BN ** 2 - 1 / A_inD ** 2) # Eq. (B.44)
        r_B = (3 * V_B / (4 * np.pi)) ** (1/3)
        A_B = 4 * np.pi * r_B ** 2 # Mentioned below Eq. (B.43)
        dp = self.bladder_args['C_p'] * p_BN * ((A_BN - A_B) / A_BN) ** 2 if self.bladder_args['tissue_pressed_in_BN'] else 0 # Eq. (B.43)

        p_T = p_D - (self.bladder_args['rho'] * Q2) / (2 * A_T ** 2) - dp # Eq. (B.42)
        return p_T - p_S # Eq. (B.53)

    def f1(self, V_B, f_aD_s, Q):
        self.p_D = self.get_p_D(V_B, f_aD_s, Q)
        return self.Q_in - Q # Eq. (3)

    def f2(self, f_aD_s, w_e_s, w_i_s):
        return 1 / self.bladder_args['tau_D'] * (w_e_s- f_aD_s - w_i_s * f_aD_s) # df_aD_s - Eq. (1)

    def f3(self, f_aS_s, w_s_s):
        return (1 / self.bladder_args['tau_S']) * (w_s_s - f_aS_s) # df_aS_s - Eq. (2)

    def fmap(self, V_B, f_aD_s, f_aS_s):
        try:
            r_U = optimize.bisect(lambda r_U: self.f_0(V_B, f_aD_s, f_aS_s, r_U), 0, 5 * 10 ** -3)
        except ValueError:
            r_U = 0
        return r_U
    
    def sigma_u_s(self, u_s):
        if u_s < 0:
            sigma = 1.8 - (0.8 * (1 + u_s))/(1 - 7.56 * u_s / self.bladder_args['k'])
        elif u_s == 0:
            sigma = 1
        else:
            sigma = (1 - u_s)/(1 + (u_s / self.bladder_args['k']))
        return sigma

    def sigma_upasD(self, u_D_s):
        return self.bladder_args['C_u'] * u_D_s # Eq. (B.34)

    def sigma_lD_s(self, l_D):
        l_D_s = l_D / self.bladder_args['l_optD']
        if l_D_s <= 0.35:
            sigma = 0
        elif l_D_s <= 0.45:
            sigma = 5.5 * l_D_s - 1.925
        elif l_D_s <= 1.1:
            sigma = 0.643 * l_D_s + 0.293
        elif l_D_s <= 1.4:
            sigma = -3.33333 * l_D_s + 4.66667
        else:
            sigma = 0
        return sigma

    def line_eq(self, x1, y1, x2, y2):
        "Find equation of line between two (x,y) points"
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        return m, c

    def sigma_lS_s(self, l_S):
        l_S_s = l_S / self.bladder_args['l_optS']
        lower_l_S, upper_l_S = 1.73 * 10 ** -6, 2.89 * 10 ** -6
        lower_l_S_s, upper_l_S_s = lower_l_S / self.bladder_args['l_optS'], upper_l_S / self.bladder_args['l_optS']
        points = [(0.55, 0), (lower_l_S_s, 0.8), (1, 1), (1.1, 1), (1.75, 0)]
        for i in range(len(points) - 1):
            if l_S_s >= points[i][0] and l_S_s < points[i + 1][0]:
                m, c = self.line_eq(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                sigma = m * l_S_s + c
                return sigma
        return 0

    def get_p_D(self, V_B, f_aD_s, Q):
        r_outD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD'] + self.bladder_args['V_muscleD']) ) ** (1/3) # Eq. (B.3)
        r_inD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD']) ) ** (1/3) # Eq. (B.4)
        r_D = (r_outD + r_inD) / 2 # Eq. (B.2)

        u_D = (Q / (8 * np.pi * self.bladder_args['r_optD'])) * (1 / r_outD ** 2 + 1 / r_inD ** 2) # Eq. (B.22), assume Q_in = 0 as mentioned in B.37
        self.u_D = u_D #Save this so I can analyse it. 

        u_D_s = u_D / self.bladder_args['u_maxD'] # Mentioned in paragraph above B.3.3
        sigma_uD_s = self.sigma_u_s(u_D_s) # Eq. (B.31)

        l_D = (self.bladder_args['l_optD'] / self.bladder_args['r_optD']) * r_D # Eq. (B.5)

        sigma_lpasD = 0 if r_D < self.bladder_args['r_0D'] else self.bladder_args['C_l'] * ((r_D - self.bladder_args['r_0D']) / self.bladder_args['r_0D']) ** self.bladder_args['alpha'] # Eq. (B.32)
        
        sigma_nomD = f_aD_s * self.bladder_args['sigma_isoD'] * sigma_uD_s * self.sigma_lD_s(l_D) + sigma_lpasD + self.sigma_upasD(u_D_s) # Eq. (4). Active + passive elastic + passive viscoelastic tensile stress
        A_D = np.pi * (r_outD ** 2 - r_inD ** 2) # Eq. (5)
        sigma_D = (self.bladder_args['A_nomD'] / A_D) * sigma_nomD # Eq. (6)
        p_D = sigma_D * np.log(r_outD / r_inD) # Eq. (10)
        
        if p_D < 0:
            p_D = 0
        
        return p_D

    def get_Q(self, f_aS_s, r_U):
        A_T = np.pi * (r_U + self.bladder_args['dr']) ** 2 # Eq. (B.15)
        A_U = np.pi * r_U ** 2 # Eq. (B.8)
        p_S = self.get_p_S(A_U, f_aS_s, r_U)
        self.p_S = p_S
        Q2 = self.get_Q2(A_U, A_T, p_S)
        return np.sqrt(Q2)

    #Function that updates hypogastric input - aka w_i_s
    #Here modified to allow integration with neural sim.
    def update_sympathetic_input(self):
        w=self.w_i_s
        return w
    
    
    #Function that updates pelvic input - aka w_e_s
    def update_parasympathetic_input(self):
        w=self.w_e_s
        return w

    #Function that updates pudendal input - aka w_s_s
    def update_somatic_input(self):
        w=self.w_s_s
        return w
    
    #Function that determines if voiding state is true/false
    def is_voiding(self):
        if self.voiding and self.V_B < 1 * 10 ** -8:
            return False # Reset voiding if bladder is empty
        elif self.p_D >= self.bladder_args['p_theta']:
            return True # If pressure exceeds threshold, voiding
        else:
            return self.voiding # Otherwise, maintain voiding state


    #Main function, takes paramters for bladder input, and outputs biophysical parameters
    def process_neural_input(self, w_e_s, w_i_s, w_s_s, maxTime, dT):
        self.Q_in = self.bladder_args['C_Qin']
        self.w_e_s, self.w_i_s, self.w_s_s = w_e_s, w_i_s, w_s_s
        datadict = [{'V_B': self.V_B, 'f_aD_s': self.f_aD_s, 'f_aS_s': self.f_aS_s, 'r_U': self.r_U, 'Q': self.Q, 'p_D': self.p_D, 'p_S': self.p_S, 'Q_in': self.Q_in}]
        self.bladder_args['dT'] = dT
        
        ts = np.arange(0, maxTime, dT)

        for stamp, t in enumerate(ts):
            self.timestamp = stamp #the step of the simulation (for indexing external data)
            self.t = t #the actual time of the simulation
            self.voiding = self.is_voiding()
            d_w_e_s = self.update_parasympathetic_input()
            d_w_i_s = self.update_sympathetic_input()
            d_w_s_s = self.update_somatic_input()

            self.w_e_s = d_w_e_s
            self.w_i_s = d_w_i_s
            self.w_s_s = d_w_s_s
            
            self.f_aD_s += dT * self.f2(self.f_aD_s, self.w_e_s, self.w_i_s)
            self.f_aS_s += dT * self.f3(self.f_aS_s, self.w_s_s)
            self.Q_in = self.get_Qin(t)
            self.V_B += dT * self.f1(self.V_B, self.f_aD_s, self.Q)
            self.V_B = min(max(self.V_B, 0), 5 * 10 ** -4)
            self.r_U = self.fmap(self.V_B, self.f_aD_s, self.f_aS_s)
            self.Q = self.get_Q(self.f_aS_s, self.r_U)

            datadict.append({'t': t, 'V_B': self.V_B, 'f_aD_s': self.f_aD_s, 'f_aS_s': self.f_aS_s, 'r_U': self.r_U, 'Q': self.Q, 'p_D': self.p_D, 'p_S': self.p_S, 'Q_in': self.Q_in, 'w_e_s': self.w_e_s, 'w_i_s': self.w_i_s, 'w_s_s': self.w_s_s, 'voiding': self.voiding})

        df = pd.DataFrame(datadict)
        return df

#-----------------------------------------------------------
"""
Now shall define the main simulation function
"""
#-----------------------------------------------------------

def bladder_sim(
        time,
        tibial_parameters,
        parameters, 
        pag = True,
        pmc = True,
        spine_aff = True,
        random_gen = False,
        seed = 1
):

    """
    This is the main function to be called when running a simulation of the bladder and neural network.

    Arguments:
    ------------

    time (float): The duration (in seconds) of bladder and circuit behaviour that should be produced.

    tibial_parameters (list): The tibial nerve stimulation that is to be applied to the system, formatted as [frequency(Hz), duration(s)] e.g. [1, 1000]

    parameters (dataframe): The parameters specified in the readme, dictating the weights of the synaptic connections within the model. Imported from params.py 
                            "from params import parameters".

    pag (bool, default True): If tibial nerve projections to the Periaquaductal Grey (PAG) should be maintained (True/False)

    pmc (bool, default True): If tibial nerve projections to the Pontine Micturition Centre (PMC) should be maintained (True/False)
     
    spine_aff (bool, default True): If tibial nerve projections to spinal afferents should be maintained (True/False)
    
    random_gen (bool, default False): If the simulation should be provided a seed (i.e., returning the same results for any noise calculation). 
                                  NOTE: If this is set to true and supplied a seed the simulation will act in a non-random manner!

    rand_seed (int, default 1): Seed for any random noise calculation 

    Returns:
    -------------

    results (list): A list of key results from the simulation, formatted as so:

    +-------------------------------+------------------------------------------------------------------------------------------------------------+
    |           Variable            |                                                 Definition                                                 |
    +-------------------------------+------------------------------------------------------------------------------------------------------------+
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
    +-------------------------------+------------------------------------------------------------------------------------------------------------+
    """
    ###########Neuronal Network Definition#################
    # Initiate neuronal network related variables
    # Import weights from before
    runtime=time
    params = parameters["parameters"] #Model weights (externally supplied) 
    min_wgt = parameters["min_weights"] #Lower bound of the synaptic weights 
    max_wgt = parameters["max_weights"] #Upper bound of the synaptic weights

    cores = 1 #Number of cores to use when running simulation (Defaults to one unless otherwise specified)
    tibial_params = tibial_parameters #tibial stimulation (rate, duration)

    monitors = None #Variable to hold sim monitors after network construction. 
    results = None #Hold results of simulation
    
    global model
    model=LUT() #define outside of function, to store variables
    

    # Define function to smooth firing rate
    def fr_smooth(monitor, crit_freq, sampling_freq):
        """
        Function takes raw firing rate data from a PopulationFiringRate class (Brian2) 
        and filters/smooths it to return a continuous average firing rate for the population.

        Function uses a 1st order Butterworth filter to smooth the data.
        
        Arguments:
        -----------
        monitor : PopulationRateMonitor object,
            The monitor containing the raw instantaneous firing rates recorded from the target population.
            NOTE: Must be supplied AFTER running a simulation.
        
        crit_freq : float,
            The desired cutoff frequency for the filter.
        
        sampling_frequ : float,
            The sampling frequency of the monitor data (Usuall default clock time for simulation).

        """

        # Import monitor and extract raw data
        dat = monitor.get_states()["rate"]

        # Define filter numerator and denomenator
        sos = butter(1, Wn=crit_freq, fs=sampling_freq, output="sos", analog=False)

        # Apply the filter to the data
        return sosfilt(sos, dat)



    ###################################Experimental Parameters###################################

    # Build the device preferences, specifying any multicore processing required
    # Will run the simulation in standalone mode.
    device.reinit()
    device.activate()

    # Define default clock cycle (50Hz, or 20ms per step)
    defaultclock.dt = 20*ms 
    
    if random_gen is True:
        # Specify seed for pseudorandom number generation (For testing/plotting only, disable this if running anlysis!)
        device.seed(seed=rand_seed) #Keep for reproduceable raster images!

    # Specify experimental parameters
    update_time = 20 #The rate at which the timedarrays shall be updated (Hz)

    # Tibial connectivity
    PMC_connected = pmc #Does the TN connect to PMC
    PAG_connected = pag #Does the TN connect to the PAG
    Spine_connected = spine_aff #Does the TN connect to the spine? 

    # Tibial Strength.

    tib_weight = [2, 4] #Weight of tibial connections (spine, brain), fit from data. 

    tib_rate = tibial_params[0] #frequency of tibial input (Hz) 
    tib_dur = tibial_params[1] #duration of tibial input (s)

    ###################################Define Model Parameters###################################
    # Dynamics of a basic adaptive firing neuron

    # Membrane dynamics determined by Gorski et al.
    Cm = 200*pF #Membrane capacitance, relatively similar across neurons
    E_A = -70*mV #Reversal potential of adaption conductance (outward current that drives hyperpolarisation)
    E_L = -60*mV #Resting potential (reversal potential of leak current)
    E_L_tonic = -70*mV 

    g_L = 10*nS #Resting conductance

    v_reset = -55*mV #Post-spike reset potential
    v_th = -50*mV #Spike threshold

    delta_t = 10*mV #Slope of the spike initiation (i.e, depolarisation)
    delta_t_ref = 5*ms #refractory period

    # Subthreshold adaption parameters
    v_A = -50*mV #Activation threshold for sub-spike adaption (drives changing firing rate)
    v_A_tonic = -45*mV 

    delta_A = 5*mV #Slope of subthreshold activation (severity of spike rate adaption)
    delta_g_A = 1*nS #Stepwise increase in adaption current (set to zero for non-adaption)
    delta_g_A_tonic = 0*nS

    g_A_max = params[45]*nS
    g_A_max_tonic = 2*nS 

    tau_A = 200*msecond #Decay rate for adaption current
    tau_A_tonic = 40*msecond 

    # Tonic firing parameter (causes self sustained firing)
    Iap = params[44]*pA

    # Synaptic parameters
    E_ex = 0*mV #Reversal potential of excitatory current
    E_in = -80*mV #Reversal potential of inhibitory current

    # Opioidergic parameters (To be fit!)
    E_op = -80*mV #Reversal potential of enkephalinergic current 
    tau_op = 10*ms #Rate of enkephalinergic conductance decay, short term lingering inhibition.

    g_op_increment = 1.5*nS #Baseline conductance of enkephalinergic synapse

    ##Learning parameters, determined by Vogels et. al., 2011

    # First decay constants (For conductance)
    tau_glut = 5*ms #Excitatory rate of conductance decay (Glutamatergic transmission)
    tau_gaba = 10*ms #Inhibitory rate of conductance decay (GABAergic transmission)
    tau_stdp = 20*ms #Rate of memory trace decay

    # Then weight increments (To be fit)
    g_ex_increment = params[42]*nS #Baseline excitatory conductance

    g_in_increment = params[43]*nS #Baseline inhibitory conductance

    # Max weights
    ex_w_max = 100 # Max excitatory weight

    in_w_max = 100 #Max inhibitory weigth

    # Learning rates and depression parameters
    learn_rate = 5*10**-3 #Learning rate of synapse

    p_o = 1*Hz #Target postsynaptic firing rate

    dep_fac = 2*tau_stdp*p_o #Rate of synaptic depression

    # Population parameter
    pop = 100 #Number of neurons in each unit, NOTE: Be careful when editing this, it is memory intensive!

    # Initial bladder simulation parameters
    global w_e_s
    global w_i_s
    global w_s_s

    w_e_s = 0
    w_i_s = 0
    w_s_s = 0

    ############################################################################################

    # Define Model Equations
    # First, set equation for tonically active (nonadaptive) neurons within the circuit
    # These neurons possess both an external current (Iap) and synaptic input (Is) for modulation
    # Iap is presumed to be constant

    tonic_eqs = """
    dv/dt = (g_L*(E_L_tonic - v) + g_L*delta_t*exp((v - v_th)/delta_t) + g_A*(E_A - v) + Iap + Is) / Cm : volt (unless refractory)
    dg_A/dt = ((g_A_max_tonic / (1 + exp((v_A_tonic - v)/delta_A))) - g_A) / tau_A_tonic : siemens
    Is = g_ex*(E_ex - v) + g_in*(E_in - v) + g_op*(E_op - v): amp
    dg_ex/dt = -g_ex / tau_glut : siemens
    dg_in/dt = -g_in / tau_gaba : siemens
    dg_op/dt = -g_op / tau_op : siemens
    """

    # Then, define equation for normal adaptive neurons
    # NOTE: the rate of adaption is determined by delta_g_A (the change in adaption with each spike)
    adapt_eqs = """
    dv/dt = (g_L*(E_L - v) + g_L*delta_t*exp((v - v_th)/delta_t) + g_A*(E_A - v) + Is) / Cm : volt (unless refractory)
    dg_A/dt = ((g_A_max / (1 + exp((v_A - v)/delta_A))) - g_A) / tau_A : siemens
    Is = g_ex*(E_ex - v) + g_in*(E_in - v) + g_op*(E_op - v): amp
    dg_ex/dt = -g_ex / tau_glut : siemens
    dg_in/dt = -g_in / tau_gaba : siemens
    dg_op/dt = -g_op / tau_op : siemens
    """

    #####################################Synaptic Equations###################################
    """
    NOTE: Must clip weight within a range of 0 <= W <= ex_w_max
    NOTE: Weight change however, has no limit! 
    """

    # Define effects of subthreshold adaption
    # Delta_g_A determines rate of adaption
    on_spike = """
    v = v_reset 
    g_A += delta_g_A
    """

    on_spike_tonic = """
    v = v_reset 
    g_A += delta_g_A_tonic
    """

    # Define synaptic model to allow for learning
    # Importantly, this constrains the weight of each connection to a range specified when the sim is built.
    # NOTE: a_pr & a_pst are synaptic traces (memory)
    # NOTE: traces updated only on spiking, to save memory

    syn_model = """
    max_weight : 1
    min_weight : 1
    weight : 1
    da_pr/dt = -a_pr/tau_stdp : 1 (event-driven)
    da_pst/dt = -a_pst/tau_stdp : 1 (event-driven)
    """

    # Define effects of presynaptic spiking
    # First for excitatory synapses
    on_pre_ex = """
    a_pr += 1 #Increase presynaptic memory trace
    weight = clip(weight + (learn_rate * (a_pst - dep_fac)), min_weight, max_weight) #Increment the weight based on postsynaptic activity
    g_ex += g_ex_increment * weight #Increase conductance representing transmission
    """

    # Then for inhibitory
    on_pre_in = """
    a_pr += 1 #Increase presynaptic memory trace
    weight = clip(weight + (learn_rate * (a_pst - dep_fac)), min_weight, max_weight) #Increment the weight based on postsynaptic activity
    g_in += g_in_increment * weight #Increase conductance representing transmission
    """

    # Define effects of postsynaptic spiking
    # First for excitatory synapses
    on_post_ex = """
    a_pst += 1 #Increase postsynaptic memory trace
    weight = clip(weight + (learn_rate * a_pr), min_weight, max_weight) #Increment weight based on presynaptic activity
    """

    # Then inhibitory
    on_post_in = """
    a_pst += 1 #Increase postsynaptic memory trace
    weight = clip(weight + (learn_rate * a_pr), min_weight, max_weight) #Increment weight based on presynaptic activity
    """

    # Finally, set up new equations for opioidergic transmission (inhibitory)
    on_pre_op="""
    a_pr += 1 #Increase memory trace (STDP)
    weight = clip(weight + (learn_rate * (a_pst - dep_fac)), min_weight, max_weight)
    g_op += g_op_increment * weight
    """

    on_post_op="""
    a_pst += 1
    weight = clip(weight + (learn_rate * a_pr), min_weight, max_weight)
    """

    ##################################BLADDER#################################################
    # First, led the bladder be defined as a single neuronal unit with no set equation
    # Instead with a series of dimensionless parameters:
    # Pressure (Pb)
    # Volume (V)

    ######Define Bladder Function###########
    bladder_equations = """
    pressure : 1
    volume : 1
    f_aD_s_val : 1
    f_aS_s_val : 1
    r_U_val : 1
    Q_val : 1
    p_S_val : 1
    Q_in_val : 1
    w_e_s_val : 1
    w_i_s_val : 1
    w_s_s_val : 1
    voiding_state : 1
    u_D_val : 1
    """  

    global bladder
    bladder = NeuronGroup(1, model=bladder_equations)
    
    global bladder_pressure_mon
    bladder_pressure_mon = StateMonitor(bladder, "pressure", record=True, when="end")
    
    global bladder_volume_mon
    bladder_volume_mon = StateMonitor(bladder, "volume", record=True, when="end")
    
    global bladder_secondary_state_mon
    bladder_secondary_state_mon = StateMonitor(
        bladder,
        [
            "f_aD_s_val",
            "f_aS_s_val",
            "r_U_val",
            "Q_val",
            "p_S_val",
            "Q_in_val",
            "w_e_s_val",
            "w_i_s_val",
            "w_s_s_val",
            "voiding_state",
            "u_D_val"
        ],
        record=True,
        when="end",
    )


    @network_operation(when="start")
    def bladder_model_run():
        """
        This function runs the bladder sim calculation at each timestep. 
        Updates the model object, which allows other functions to take use of it. 
        """
        global w_e_s
        global w_i_s
        global w_s_s
        global model
        global bladder

        #Run bladder model and extract pressure/volume
        model_state=model.process_neural_input(w_e_s, w_i_s, w_s_s, 0.2, 0.2)
        volume_state=model_state["V_B"].tail(1)
        pressure_state = model_state["p_D"].tail(1)

        #Update bladder states at each timestep
        bladder.pressure[:] = pressure_state[1]
        bladder.volume[:] = volume_state[1]

        #Other less important parameters
        bladder.f_aD_s_val[:] = model_state["f_aD_s"].tail(1)[1]
        bladder.f_aS_s_val[:] = model_state["f_aS_s"].tail(1)[1]
        bladder.r_U_val[:] = model_state["r_U"].tail(1)[1]
        bladder.Q_val[:] = model_state["Q"].tail(1)[1]
        bladder.p_S_val[:] = model_state["p_S"].tail(1)[1]
        bladder.Q_in_val[:] = model_state["Q_in"].tail(1)[1]
        bladder.w_e_s_val[:] = model_state["w_e_s"].tail(1)[1]
        bladder.w_i_s_val[:] = model_state["w_i_s"].tail(1)[1]
        bladder.w_s_s_val[:] = model_state["w_s_s"].tail(1)[1]
        bladder.voiding_state[:] = model_state["voiding"].tail(1)[1]
        bladder.u_D_val[:] = model.u_D


    # -------------------Calculate Afferent Parameters---------------------

    # Generate a function, that calculates the afferent rate based on current bladder pressure.
    afferent_rate = [0]
    @network_operation(when="start") #Run before all other calculations performed.
    def afferent_rate_calculation():
        #Get value of bladder pressure for time t
        global bladder_pressure_mon
        global afferent_rate

        #Set to zero for first timestep
        if bladder_pressure_mon.pressure.size == 0:
            current_pressure=0
        
        else:
            current_pressure = bladder_pressure_mon.pressure[0][-1]

        #Calculate the rate for a given bladder pressure From McGee et al. 
        afferent_rate = (-3*10**-8)*current_pressure**5 + (1*10**-5)*current_pressure**4 - (1.5*10**-3)*current_pressure**3 + (7.9*10**-2)*current_pressure**2 - 0.6*current_pressure

    # Calculate desired tibial nerve input
    # Create array of zeroes length of bladder data
    tib_dat = np.zeros(runtime)

    # Replace N values with rate
    num_cycles = int(tib_dur/(update_time/1000)) #convert from seconds to clock cycles (divide by s)
    tib_dat[0:num_cycles] = tib_rate

    # Convert to brian-readable timed array.
    in_input = TimedArray(tib_dat*Hz, dt=update_time*ms)

    # -----------------------------Define Bladder and Afferents------------------

    # Define the internal urethral sphincter

    global ius
    ius = PoissonGroup(1, rates=0*Hz, name="IUS")
    ius_mon = PopulationRateMonitor(ius)

    global eus
    eus = PoissonGroup(1, rates=0*Hz, name="EUS")
    eus_mon = PopulationRateMonitor(eus)

    @network_operation
    def update_ius(when="start"):
        #Set the rate of the afferent to
        global bladder_secondary_state_mon
        global ius

        if bladder_secondary_state_mon.f_aS_s_val.size == 0:
            ius.rates=0*Hz
        
        else:
            ius.rates=bladder_secondary_state_mon.f_aS_s_val[0][-1]*Hz
        
    @network_operation
    def update_eus(when="start"):
        #Set the rate of the afferent
        global bladder_secondary_state_mon
        global eus

        if bladder_secondary_state_mon.f_aS_s_val.size == 0:
            eus.rates=0*Hz
        
        else:
            eus.rates=bladder_secondary_state_mon.f_aS_s_val[0][-1]*Hz

    ##################################Neuronal Group Definition#####################################

    # Tibial Nerve
    # -----------------------------------------------------------------------
    tibial_nerve = PoissonGroup(pop, rates="in_input(t)", name="TibialNerve")

    # Spinal Interneuron (GABAergic)
    tibial_classic = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    tibial_classic.v = "E_L + rand()*(v_th - E_L)"

    # Ascending Opioidergic
    tibial_op = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    tibial_op.v = "E_L + rand()*(v_th - E_L)"

    # Poisson Neurons
    # -----------------------------------------------------------------------
    # The Pelvic Nerve (NOTE: This is a poisson group whose firing rate is determined by Pb!)
    # Updates its firing rate with each timestep (starts at zero Hz at t=0)

    global pelvic_aff
    pelvic_aff = PoissonGroup(pop, rates=0*Hz, name="PelvicAfferent")
    # -----------------------------------------------------------------------

    # Define a function to update the pelvic afferent firing rate for each step of the simulation
    @network_operation
    def update_afferent(when="start"):
        #Set the rate of the afferent to
        global pelvic_aff
        global afferent_rate
        pelvic_aff.rates=afferent_rate*Hz


    ##Brainstem Neurons##

    # First, Independent tonic neurons
    tonic_ind_pop = pop
    tonic_ind = NeuronGroup(pop, tonic_eqs, threshold="v>0*mV", reset=on_spike_tonic, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup")
    tonic_ind.v = "E_L + rand()*(v_th - E_L)"

    tonic_ind_PMC_pop = pop
    tonic_ind_PMC = NeuronGroup(pop, tonic_eqs, threshold="v>0*mV", reset=on_spike_tonic, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup2")
    tonic_ind_PMC.v = "E_L + rand()*(v_th - E_L)"

    # Then inverse neurons
    inv_neuron_pop = pop
    inv_neuron = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup3")
    inv_neuron.v = "E_L + rand()*(v_th - E_L)"

    # Type I direct neurons (RENAMED TO des_in from t1_direct)
    des_in_pop = pop
    des_in = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup4")
    des_in.v = "E_L + rand()*(v_th - E_L)"

    # Type II direct neurons
    t2_direct_pop = pop
    t2_direct = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    t2_direct.v = "E_L + rand()*(v_th - E_L)"

    # Transient neurons
    transient_pop = pop
    transient = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    transient.v = "E_L + rand()*(v_th - E_L)"

    # Finally, Relay neurons
    # First pathway A (upper) relay
    relay_A_pop = pop
    relay_A = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    relay_A.v = "E_L + rand()*(v_th - E_L)"

    # Then pathway B (lower) relay
    relay_B_pop = pop
    relay_B = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    relay_B.v = "E_L + rand()*(v_th - E_L)"

    ##Pudendal Loop##
    # Pudendal motorneurons (Onuf's Nucleus)
    onuf = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    onuf.v = "E_L + rand()*(v_th - E_L)"

    # Preganglionic neurons within the spine
    pgn = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pgn.v = "E_L + rand()*(v_th - E_L)"

    # Pudendal afferent
    pud_aff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pud_aff.v = "E_L + rand()*(v_th - E_L)"

    # Pudendal Process
    pud_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pud_proc.v = "E_L + rand()*(v_th - E_L)"

    # Circuit connectivity
    pgn_stim = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pgn_stim.v = "E_L + rand()*(v_th - E_L)"

    # Then Circuit IO
    # Ascending output
    asc_out = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    asc_out.v = "E_L + rand()*(v_th - E_L)"

    # Onuf inhibition
    onuf_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    onuf_inhib.v = "E_L + rand()*(v_th - E_L)"

    ##Hypogastric loop##
    # First major hypogastric neuron groups

    hyp_nerve = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    hyp_nerve.v = "E_L + rand()*(v_th - E_L)"

    img = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    img.v = "E_L + rand()*(v_th - E_L)"

    hyp_eff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    hyp_eff.v = "E_L + rand()*(v_th - E_L)"

    # Descending inhibition
    hyp_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    hyp_inhib.v = "E_L + rand()*(v_th - E_L)"

    ##Pelvic Loop##
    # The last reflex loop to be constructed is the connections within the pelvic nerve
    # First major neuron groups

    ####################
    # Defined new pelvic intermediate.
    pelvic_gang = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pelvic_gang.v = "E_L + rand()*(v_th - E_L)"

    pelvic_eff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pelvic_eff.v = "E_L + rand()*(v_th - E_L)"

    pelvic_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pelvic_proc.v = "E_L + rand()*(v_th - E_L)"

    # Including other connections
    hyp_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    hyp_proc.v = "E_L + rand()*(v_th - E_L)"

    ##Remaining PGN Connections##
    # Inhibitory neuron driven by PGN
    pgn_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pgn_inhib.v = "E_L + rand()*(v_th - E_L)"

    # Inverse neuron that inhibits PGN via descending input
    des_inverse = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    des_inverse.v = "E_L + rand()*(v_th - E_L)"

    # PGN reciprocal inhibitor
    pgn_reciprocal = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
    pgn_reciprocal.v = "E_L + rand()*(v_th - E_L)"

    ##################################Synapses#################################################

    ##Tibial Nerve
    # ----------------------------------------------------------------------

    # First from Tibial nerve to two interneurons (Excitatory, classical)

    tib_tib_classic = Synapses(tibial_nerve, tibial_classic, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
    tib_tib_classic.connect(condition='i!=j', p = 1)
    tib_tib_classic.weight = tib_weight[0]
    tib_tib_classic.min_weight = tib_weight[0]
    tib_tib_classic.max_weight = tib_weight[0]

    tib_tib_op = Synapses(tibial_nerve, tibial_op, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
    tib_tib_op.connect(condition='i!=j', p = 1)
    tib_tib_op.weight = tib_weight[1]
    tib_tib_op.max_weight = tib_weight[1]
    tib_tib_op.min_weight = tib_weight[1]

    if Spine_connected is True:
        # Then from classical spinal branch to ascending output (Inhibitory, classical)
        tib_classic_asc_out = Synapses(tibial_classic, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)
        tib_classic_asc_out.connect(condition='i!=j', p = 1)
        tib_classic_asc_out.weight = tib_weight[0]
        tib_classic_asc_out.max_weight = tib_weight[0]
        tib_classic_asc_out.min_weight = tib_weight[0]

    if PMC_connected is True:
        # If this value is true then the tibial nerve will project to connections in the PMC
        # To transient neurons
        tib_op_transient = Synapses(tibial_op, transient, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_transient.connect(condition='i!=j', p = 1)
        tib_op_transient.weight = tib_weight[1]
        tib_op_transient.max_weight = tib_weight[1]
        tib_op_transient.min_weight = tib_weight[1]

        # To inverse neuron (inhibitory, enkephalinergic)
        tib_op_in = Synapses(tibial_op, inv_neuron, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_in.connect(condition='i!=j', p = 1)
        tib_op_in.weight = tib_weight[1]
        tib_op_in.max_weight = tib_weight[1]
        tib_op_in.min_weight = tib_weight[1]

        # To T2 direct (feedback)
        tib_op_t2 = Synapses(tibial_op, t2_direct, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_t2.connect(condition='i!=j', p = 1)
        tib_op_t2.weight = tib_weight[1]
        tib_op_t2.max_weight = tib_weight[1]
        tib_op_t2.min_weight = tib_weight[1]

        # To tonic independent
        tib_op_tonic_ind = Synapses(tibial_op, tonic_ind_PMC, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_tonic_ind.connect(condition='i!=j', p = 1)
        tib_op_tonic_ind.weight = tib_weight[1]
        tib_op_tonic_ind.max_weight = tib_weight[1]
        tib_op_tonic_ind.min_weight = tib_weight[1]

        # To type I direct (des input)
        tib_op_des_in = Synapses(tibial_op, des_in, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_des_in.connect(condition='i!=j', p = 1)
        tib_op_des_in.weight = tib_weight[1]
        tib_op_des_in.max_weight = tib_weight[1]
        tib_op_des_in.min_weight = tib_weight[1]

    if PAG_connected is True:
        # If this value is true, then the tibial nerve will project to the PAG of the brainstem
        # To the upper relay
        tib_op_relay_A = Synapses(tibial_op, relay_A, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_relay_A.connect(condition='i!=j', p = 1)
        tib_op_relay_A.weight = tib_weight[1]
        tib_op_relay_A.max_weight = tib_weight[1]
        tib_op_relay_A.min_weight = tib_weight[1]

        # To the lower relay
        tib_op_relay_B = Synapses(tibial_op, relay_B, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_relay_B.connect(condition='i!=j', p = 1)
        tib_op_relay_B.weight = tib_weight[1]
        tib_op_relay_B.max_weight = tib_weight[1]
        tib_op_relay_B.min_weight = tib_weight[1]

        # To the tonic independent neuron
        tib_op_tonic_ind_pag = Synapses(tibial_op, tonic_ind, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
        tib_op_tonic_ind_pag.connect(condition='i!=j', p = 1)
        tib_op_tonic_ind_pag.weight = tib_weight[1]
        tib_op_tonic_ind_pag.max_weight = tib_weight[1]
        tib_op_tonic_ind_pag.min_weight = tib_weight[1]

    # ------------------------------------------------------------------------------
    ###Brainstem Switch###
    ##PAG GROUPS##
    # First inhibitory connection from Independant -> Relay in PAG
    pag_I_R = Synapses(tonic_ind, relay_A, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)
    pag_I_R.connect(condition='i!=j', p = 1)
    pag_I_R.weight = params[0]
    pag_I_R.max_weight = max_wgt[0]
    pag_I_R.min_weight = min_wgt[0]

    # Then excitatory connection from Relay in PAG to Transient neuron & Type I direct
    # First to transient
    R_Tr = Synapses(relay_A, transient, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    R_Tr.connect(condition='i!=j', p = 1)
    R_Tr.weight = params[1]
    R_Tr.max_weight = max_wgt[1]
    R_Tr.min_weight = min_wgt[1]

    # From secondary relay (pathway B) to inverse neuron
    R2_In = Synapses(relay_B, inv_neuron, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    R2_In.connect(condition='i!=j', p = 1)
    R2_In.weight = params[2]
    R2_In.max_weight = max_wgt[2]
    R2_In.min_weight = min_wgt[2]

    # Then PAG relay to type I direct neuron
    R_Di = Synapses(relay_A, des_in, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    R_Di.connect(condition='i!=j', p = 1)
    R_Di.weight = params[3]
    R_Di.max_weight = max_wgt[3]
    R_Di.min_weight = min_wgt[3]

    # Then input from spine to relay 1 (excitatory)
    asc_out_R = Synapses(asc_out, relay_A, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    asc_out_R.connect(condition='i!=j', p = 1)
    asc_out_R.weight = params[4]
    asc_out_R.max_weight = max_wgt[4]
    asc_out_R.min_weight = min_wgt[4]

    # And input from spine to relay 2 (excitatory)
    asc_out_R2 = Synapses(asc_out, relay_B, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    asc_out_R2.connect(condition='i!=j', p = 1)
    asc_out_R2.weight = params[5]
    asc_out_R2.max_weight = max_wgt[5]
    asc_out_R2.min_weight = min_wgt[5]

    ##PMC GROUPS##
    # Then Transient to Inverse neuron
    Tr_In = Synapses(transient, inv_neuron, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    Tr_In.connect(condition='i!=j', p = 1)
    Tr_In.weight = params[6]
    Tr_In.max_weight = max_wgt[6]
    Tr_In.min_weight = min_wgt[6]

    # Then from type I direct to type II direct
    Di_Dii = Synapses(des_in, t2_direct, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    Di_Dii.connect(condition='i!=j', p = 1)
    Di_Dii.weight = params[7]
    Di_Dii.max_weight = max_wgt[7]
    Di_Dii.min_weight = min_wgt[7]

    # Fron type II direct to inverse neuron (Positive feedback)
    Dii_In = Synapses(t2_direct, inv_neuron, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    Dii_In.connect(condition='i!=j', p = 1)
    Dii_In.weight = params[8]
    Dii_In.max_weight = max_wgt[8]
    Dii_In.min_weight = min_wgt[8]

    # Inverse neuron to type I direct
    In_Di = Synapses(inv_neuron, des_in, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    In_Di.connect(condition='i!=j', p = 1)
    In_Di.weight = params[9]
    In_Di.max_weight = max_wgt[9]
    In_Di.min_weight = min_wgt[9]

    # Finally, the second tonically active independent neuron to type I direct
    I_Di = Synapses(tonic_ind_PMC, des_in, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    I_Di.connect(condition='i!=j', p = 1)
    I_Di.weight = params[10]
    I_Di.max_weight = max_wgt[10]
    I_Di.min_weight = min_wgt[10]

    ###Pudendal Loop##
    # EUS to pudendal afferent
    eus_pud_aff = Synapses(eus, pud_aff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    eus_pud_aff.connect(condition='i!=j', p = 1)
    eus_pud_aff.weight = params[12]
    eus_pud_aff.max_weight = max_wgt[12]
    eus_pud_aff.min_weight = min_wgt[12]

    # Pudendal afferent to pudendal process
    pud_aff_pud_proc = Synapses(pud_aff, pud_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pud_aff_pud_proc.connect(condition='i!=j', p = 1)
    pud_aff_pud_proc.weight = params[13]
    pud_aff_pud_proc.max_weight = max_wgt[13]
    pud_aff_pud_proc.min_weight = min_wgt[13]

    # Pudendal process to PGN stimulator (inhibitory)
    pud_proc_pgn_stim = Synapses(pud_proc, pgn_stim, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    pud_proc_pgn_stim.connect(condition='i!=j', p = 1)
    pud_proc_pgn_stim.weight = params[14]
    pud_proc_pgn_stim.max_weight = max_wgt[14]
    pud_proc_pgn_stim.min_weight = min_wgt[14]

    # Pudendal process to PGN
    pud_proc_pgn = Synapses(pud_proc, pgn, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    pud_proc_pgn.connect(condition='i!=j', p = 1)
    pud_proc_pgn.weight = params[15]
    pud_proc_pgn.max_weight = max_wgt[15]
    pud_proc_pgn.min_weight = min_wgt[15]

    # Pudendal process to ascending output gate
    pud_proc_asc_out = Synapses(pud_proc, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    pud_proc_asc_out.connect(condition='i!=j', p = 1)
    pud_proc_asc_out.weight = params[16]
    pud_proc_asc_out.max_weight = max_wgt[16]
    pud_proc_asc_out.min_weight = min_wgt[16]

    # PGN stimulator to PGN
    pgn_stim_pgn = Synapses(pgn_stim, pgn, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pgn_stim_pgn.connect(condition='i!=j', p = 1)
    pgn_stim_pgn.weight = params[17]
    pgn_stim_pgn.max_weight = max_wgt[17]
    pgn_stim_pgn.min_weight = min_wgt[17]

    # Descending input to PGN (direct)
    des_in_pgn = Synapses(des_in, pgn, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    des_in_pgn.connect(condition='i!=j', p = 1)
    des_in_pgn.weight = params[18]
    des_in_pgn.max_weight = max_wgt[18]
    des_in_pgn.min_weight = min_wgt[18]

    # Descending input to PGN stimulator
    des_in_pgn_stim = Synapses(des_in, pgn_stim, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    des_in_pgn_stim.connect(condition='i!=j', p = 1)
    des_in_pgn_stim.weight = params[19]
    des_in_pgn_stim.max_weight = max_wgt[19]
    des_in_pgn_stim.min_weight = min_wgt[19]

    # Descending input to Onuf inhibiting interneuron
    des_in_onuf_inhib = Synapses(des_in, onuf_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    des_in_onuf_inhib.connect(condition='i!=j', p = 1)
    des_in_onuf_inhib.weight = params[20]
    des_in_onuf_inhib.max_weight = max_wgt[20]
    des_in_onuf_inhib.min_weight = min_wgt[20]

    # Onuf inhibitor to onuf's nucleus
    onuf_inhib_onuf = Synapses(onuf_inhib, onuf, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    onuf_inhib_onuf.connect(condition='i!=j', p = 1)
    onuf_inhib_onuf.weight = params[21]
    onuf_inhib_onuf.max_weight = max_wgt[21]
    onuf_inhib_onuf.min_weight = min_wgt[21]

    ##ypogastric Loop##

    # Hypogastric nerve (preganglionic) to Inferior Mesenteric Ganglia
    hyp_nerve_img = Synapses(hyp_nerve, img, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    hyp_nerve_img.connect(condition='i!=j', p = 1)
    hyp_nerve_img.weight = params[22]
    hyp_nerve_img.max_weight = max_wgt[22]
    hyp_nerve_img.min_weight = min_wgt[22]

    # IMG to hypogastric efferent
    img_hyp_eff = Synapses(img, hyp_eff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    img_hyp_eff.connect(condition='i!=j', p = 1)
    img_hyp_eff.weight = params[23]
    img_hyp_eff.max_weight = max_wgt[23]
    img_hyp_eff.min_weight = min_wgt[23]

    # ---------------------------------------------------

    ##Then descending input to inhibitory interneuron
    des_in_hyp_inhib = Synapses(des_in, hyp_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    des_in_hyp_inhib.connect(condition='i!=j', p = 1)  
    des_in_hyp_inhib.weight = params[25]                                                  
    des_in_hyp_inhib.max_weight = max_wgt[25]
    des_in_hyp_inhib.min_weight = min_wgt[25]

    # Finally, inhibitory interneuron to hypograstric nerve (preganglionic)
    hyp_inhib_hyp_nerve = Synapses(hyp_inhib, hyp_nerve, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    hyp_inhib_hyp_nerve.connect(condition='i!=j', p = 1)
    hyp_inhib_hyp_nerve.weight = params[26]
    hyp_inhib_hyp_nerve.max_weight = max_wgt[26]
    hyp_inhib_hyp_nerve.min_weight = min_wgt[26]

    # Pelvic afferent to ascending output
    pelvic_aff_asc_out = Synapses(pelvic_aff, asc_out, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pelvic_aff_asc_out.connect(condition='i!=j', p = 1)
    pelvic_aff_asc_out.weight = params[27]
    pelvic_aff_asc_out.max_weight = max_wgt[27]
    pelvic_aff_asc_out.min_weight = min_wgt[27]

    # Pelvic afferent to pelvic process
    pelvic_aff_pelvic_proc = Synapses(pelvic_aff, pelvic_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex) 

    pelvic_aff_pelvic_proc.connect(condition='i!=j', p = 1)
    pelvic_aff_pelvic_proc.weight = params[28]
    pelvic_aff_pelvic_proc.max_weight = max_wgt[28]
    pelvic_aff_pelvic_proc.min_weight = min_wgt[28]

    # Pelvic afferent to hypogastric process
    pelvic_aff_hyp_proc = Synapses(pelvic_aff, hyp_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pelvic_aff_hyp_proc.connect(condition='i!=j', p = 1)
    pelvic_aff_hyp_proc.weight = params[29]
    pelvic_aff_hyp_proc.max_weight = max_wgt[29]
    pelvic_aff_hyp_proc.min_weight = min_wgt[29]

    # Pelvic process to Onuf's Nucleus
    pelvic_proc_onuf = Synapses(pelvic_proc, onuf, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pelvic_proc_onuf.connect(condition='i!=j', p = 1)
    pelvic_proc_onuf.weight = params[30]
    pelvic_proc_onuf.max_weight = max_wgt[30]
    pelvic_proc_onuf.min_weight = min_wgt[30]

    # Hypogastric process to hypogastric nerve
    hyp_proc_hyp_nerve = Synapses(hyp_proc, hyp_nerve, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    hyp_proc_hyp_nerve.connect(condition='i!=j', p = 1)
    hyp_proc_hyp_nerve.weight = params[31]
    hyp_proc_hyp_nerve.max_weight = max_wgt[31]
    hyp_proc_hyp_nerve.min_weight = min_wgt[31]

    # PGN to pelvic ganglia
    pgn_pelvic_gang = Synapses(pgn, pelvic_gang, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
    pgn_pelvic_gang.connect(condition='i!=j', p = 1)
    pgn_pelvic_gang.weight = params[32]
    pgn_pelvic_gang.max_weight = max_wgt[32] 
    pgn_pelvic_gang.min_weight = min_wgt[32]

    # pel_ganglion to pelvic efferent
    pelvic_gang_pelvic_eff = Synapses(pelvic_gang, pelvic_eff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
    pelvic_gang_pelvic_eff.connect(condition='i!=j', p = 1)
    pelvic_gang_pelvic_eff.weight = params[33]
    pelvic_gang_pelvic_eff.max_weight = max_wgt[33]
    pelvic_gang_pelvic_eff.min_weight = min_wgt[33]

    # Hypogastric efferent (IMG) to pelvic ganglia. [As per DG&Wickens]
    img_pelvic_gang = Synapses(img, pelvic_gang, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)
    img_pelvic_gang.connect(condition='i!=j', p = 1)
    img_pelvic_gang.weight = params[34]
    img_pelvic_gang.max_weight = max_wgt[34]
    img_pelvic_gang.min_weight = min_wgt[34]


    ##Remaining Circuitry##
    # Connect new neurons to circuit
    # First pgn to inhibitory interneuron (targeting ascending gate)
    pgn_pgn_inhib = Synapses(pgn, pgn_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pgn_pgn_inhib.connect(condition='i!=j', p = 1)
    pgn_pgn_inhib.weight = params[35]
    pgn_pgn_inhib.max_weight = max_wgt[35]
    pgn_pgn_inhib.min_weight = min_wgt[35]

    # Then pgn to its reciprocal neuron
    pgn_pgn_reciprocal = Synapses(pgn, pgn_reciprocal, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    pgn_pgn_reciprocal.connect(condition='i!=j', p = 1)
    pgn_pgn_reciprocal.weight = params[36]
    pgn_pgn_reciprocal.max_weight = max_wgt[36]
    pgn_pgn_reciprocal.min_weight = min_wgt[36]

    # pgn reciprocal to pgn stimulator (inhibitory)
    pgn_reciprocal_pgn_stim = Synapses(pgn_reciprocal, pgn_stim, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    pgn_reciprocal_pgn_stim.connect(condition='i!=j', p = 1)
    pgn_reciprocal_pgn_stim.weight = params[37]
    pgn_reciprocal_pgn_stim.max_weight = max_wgt[37]
    pgn_reciprocal_pgn_stim.min_weight = min_wgt[37]

    # Then inhibitory interneuron to ascending gate (inhibitory)
    pgn_inhib_asc_out = Synapses(pgn_inhib, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    pgn_inhib_asc_out.connect(condition='i!=j', p = 1)
    pgn_inhib_asc_out.weight = params[38]
    pgn_inhib_asc_out.max_weight = max_wgt[38]
    pgn_inhib_asc_out.min_weight = min_wgt[38]

    # Descending input to inverse neuron
    des_in_des_inverse = Synapses(des_in, des_inverse, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

    des_in_des_inverse.connect(condition='i!=j', p = 1)
    des_in_des_inverse.weight = params[39]
    des_in_des_inverse.max_weight = max_wgt[39]
    des_in_des_inverse.min_weight = min_wgt[39]

    # Inverse neuron to pgn reciprocal (inhibitory)
    des_inverse_pgn_reciprocal = Synapses(des_inverse, pgn_reciprocal, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    des_inverse_pgn_reciprocal.connect(condition='i!=j', p = 1)
    des_inverse_pgn_reciprocal.weight = params[40]
    des_inverse_pgn_reciprocal.max_weight = max_wgt[40]
    des_inverse_pgn_reciprocal.min_weight = min_wgt[40]

    # Inverse neuron to inhibitory interneuron (inhibitory)
    des_inverse_pgn_inhib = Synapses(des_inverse, pgn_inhib, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

    des_inverse_pgn_inhib.connect(condition='i!=j', p = 1)
    des_inverse_pgn_inhib.weight = params[41]
    des_inverse_pgn_inhib.max_weight = max_wgt[41]
    des_inverse_pgn_inhib.min_weight = min_wgt[41]

    ##Define firing rate monitors, and specify their targets.
    # Afferents
    pelvic_aff_mon = PopulationRateMonitor(pelvic_aff) #Input to circuit from bladder

    # Efferents
    pelvic_eff_mon = PopulationRateMonitor(pelvic_eff) #Pelvic efferent Activity
    hyp_eff_mon = PopulationRateMonitor(hyp_eff)
    pud_eff_mon = PopulationRateMonitor(onuf)
    asc_mon = PopulationRateMonitor(asc_out) #Ascending second order interneurons activity. 
    des_mon = PopulationRateMonitor(des_in) #Descending output

    # Make key efferents global
    global raw_w_e_s
    global raw_w_i_s
    global raw_w_s_s
    
    raw_w_e_s = pelvic_eff_mon
    raw_w_i_s = hyp_eff_mon
    raw_w_s_s = pud_eff_mon

    # Spike Monitors
    pel_spikes = SpikeMonitor(pelvic_eff, record=[0])
    hyp_spikes = SpikeMonitor(hyp_eff, record=[0])
    pud_spikes = SpikeMonitor(onuf, record=[0])
    des_spikes = SpikeMonitor(des_in, record=[0])

    #######Define Firing Rate Functions#######
    # These functions which shall be run as a network operation aim to:
    # Extract the firing rates of key efferent projections, smooth, and export a value
    # This value shall be saved to a global variable, which will be used as part of the bladder function.
    neural_values = []

    @network_operation(when="end") #Run after all other calculations performed.
    def efferent_calculation():
        #Get value of efferent monitors for this timestep
        global raw_w_e_s
        global raw_w_i_s
        global raw_w_s_s

        #Define w values for the bladder simulation
        
        global w_e_s
        global w_i_s
        global w_s_s
        
        
        #First instance will be empty (at t=0), make this rate = 0Hz
        if not np.any(raw_w_e_s.get_states()["rate"]):
            w_e_s = 0

        else:    
            smoothed_w_e_s = raw_w_e_s.smooth_rate(width=1*second)[-1]/Hz
            w_e_s = smoothed_w_e_s
        
        if not np.any(raw_w_i_s.get_states()["rate"]):
            w_i_s = 0

        else:   
            smoothed_w_i_s = raw_w_i_s.smooth_rate(width=1*second)[-1]/Hz
            w_i_s = smoothed_w_i_s 

        if not np.any(raw_w_s_s.get_states()["rate"]):
            w_s_s = 0

        else:    
            smoothed_w_s_s = raw_w_s_s.smooth_rate(width=1*second)[-1]/Hz
            w_s_s = smoothed_w_s_s

        neural_values.append([w_e_s, w_i_s, w_s_s])

    ##########################################
    # Run Sim
    run(runtime*second, report="text", report_period=1*second)
    ##########################################

    # Return state monitors to object
    monitors = [ 
        bladder_pressure_mon.pressure[0], 
        bladder_volume_mon.volume[0],
        pelvic_aff_mon, 
        pelvic_eff_mon,
        hyp_eff_mon,
        pud_eff_mon,
        asc_mon,
        in_input,
        pel_spikes,
        hyp_spikes,
        pud_spikes, 
        des_mon,
        des_spikes
        ]
    
    pressure = monitors[0]
    volume = monitors[1]
    pel_aff = fr_smooth(monitors[2], crit_freq=1, sampling_freq=50)
    pel_out = fr_smooth(monitors[3], crit_freq=1, sampling_freq=50)
    hyp_out = fr_smooth(monitors[4], crit_freq=1, sampling_freq=50)
    pud_out = fr_smooth(monitors[5], crit_freq=1, sampling_freq=50)

    asc = fr_smooth(monitors[6], crit_freq=1, sampling_freq=50)
    asc_timestamps = monitors[6].t/ms 

    tib = monitors[7]

    results = [
        pressure, 
        volume,
        bladder_pressure_mon.t/second,
        pel_aff,
        pel_out,
        hyp_out,
        pud_out,
        asc,
        asc_timestamps,
        tib,
        pel_spikes,
        hyp_spikes,
        pud_spikes, 
        des_mon,
        des_spikes,
        bladder_secondary_state_mon
        ]
    
    return results

