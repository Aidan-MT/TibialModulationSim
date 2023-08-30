#First import required packages
#Base
from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#External data related
from scipy import io

#Smoothing
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d

#Plotting
from matplotlib.lines import Line2D

class BladderSim:
    """
    This class constructs a model of the bladder control network according to specified weights
    and then runs it based on supplied bladder data. 

    Initially, calling sim_build will only create the network and prime it to run (to save memory). 
    To actually run a simulation, sim_run must be called, this will compile the system to a C++ file and run it
    saving the results to the results attribute. 

    Simulation output may be accessed by calling the results attribute. 

    Arguments:
    ------------
    bladder_dat : path
        This is the path to a .mat file containing the bladder & neuronal data. 

    params : list
        A list containing the weights for all connections in the model, as floats.
    
    tib_pars : list
        A list containing the intensity, and duration of tibial neuromodulation on the system
        in the format [frequency, duration]
    
    cores : int
        The number of cores to use if multithreading is desired. Defaults to one. 
        NOTE: This requires C++ compiler is setup correctly! 

    Notes:
    ----------
    For the simulation to work correctly, the .mat files must be correctly organised
    Additionally, a working C++ compiler must be installed. 
    """

    def __init__(self, bladder_dat, params, tib_pars, cores=1):
        self.bladder_dat = io.loadmat(bladder_dat) #Bladder/Neural data (externally supplied)
        self.params = params #Model weights (externally supplied) 
        self.tib_pars = tib_pars #tibial stimulation (rate, duration)
        self.cores = cores #Number of cores to use when running simulation (Defaults to one unless otherwise specified)

        self.monitors = None #Variable to hold sim monitors after network construction. 
        self.results = None #Hold results of simulation

        #Construct framework
        self.sim_build()

    #Define a custom progress reporting bar to be used later
    report_func = '''
        int remaining = (int)((1-completed)/completed*elapsed+0.5);
        if (completed == 0.0)
        {
            std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s"<<std::flush;
        }
        else
        {
            int barWidth = 70;
            std::cout << "\\r[";
            int pos = barWidth * completed;
            for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
            }
            std::cout << "] " << int(completed * 100.0) << "% completed. | "<<int(remaining) <<"s remaining"<<std::flush;
        }
        '''

    #Now define function to convert this data to a useable form
    def bladder_preprocess(self, sample_rate):
        """
        Function imports MATLAB data as .mat then resamples according to desired sampling rate
        Returns two arrays: [volume, pressure]

        Arguments:
        ------------
        sample_rate : int,
                    Desired sampling rate for bladder data (Hz)
        """

        #Construct dataframe
        bladder_dat = pd.DataFrame({"label":pd.Series(self.bladder_dat).index, "list":pd.Series(self.bladder_dat).values})
        
        #Extract required data and form new dataframe
        fixed_data = []
        for k in bladder_dat.index:
            if k in [0, 1, 2, 4, 5]:
                continue
            dat1 = bladder_dat["list"].values[k].tolist()
            fixed_data.append(list(flatten(dat1)))

        #Convert list to new dataframe with correct index
        dat = pd.DataFrame(fixed_data, index=bladder_dat["label"][[3, 6, 7, 8]])

        #Performed using interpolation of data, as signal is non-periodic
        #Define interval for sampling rate
        interval = 1 / sample_rate

        #Define range of values to feed to interpolation function, filling bounds outside with zero
        press_rang = np.arange(0, dat.loc["tpressure"].max(), interval) #trial time at 1kHz
        vol_rang = np.arange(0, dat.loc["tvolume"].max(), interval)

        #Create interpolation function
        vol_func = interp1d(
            x=dat.loc["tvolume"], 
            y=dat.loc["volume"], 
            fill_value=(0, 0), 
            bounds_error=False
            )
        
        press_func = interp1d(
            x=dat.loc["tpressure"],
            y=dat.loc["pressure"],
            fill_value=(0,0),
            bounds_error=False
        )

        #Upsample data, return arrays
        upsampled_vol = vol_func(vol_rang)
        upsampled_press = press_func(press_rang)

        #Return processed/upsampled data as an array, to be converted to brian readable data. 
        return [upsampled_vol, upsampled_press]
    
    def neural_preprocess(self, sample_rate):
        """
        Function imports neuronal MATLAB data (pre sorted/smoothed) as .mat (20kHz sampling rate), 
        then resamples according to desired sampling rate (Hz).

        Returns a single array (neuronal LFP)

        Arguments:
        -----------

        sample_rate : int,
                    Desired sampling rate for bladder data (Hz)
        """
        
        #Status

        dat = pd.DataFrame({"label":pd.Series(self.bladder_dat).index, "list":pd.Series(self.bladder_dat).values})
        
        #Extract required data to form new dataframe
        fixed_data = []
        for k in dat.index:
            if k in [0, 1, 2, 3, 6, 7, 8]:
                continue
            dat1 = dat["list"].values[k].tolist()
            fixed_data.append(list(flatten(dat1)))
        
        #Convert to dataframe, transpose to resample. 
        df_unsampled = pd.DataFrame(fixed_data, index=["raw_signal", "signal_time"])

        #Create resampling function
        dnsmpl_func = interp1d(
            x=df_unsampled.loc["signal_time"],
            y=df_unsampled.loc["raw_signal"],
            fill_value=(0,0),
            bounds_error=False
        )

        #Create range to feed to function
        interval = 1 / sample_rate
        rang = np.arange(0, df_unsampled.loc["signal_time"].max(), interval)

        #Resample
        df_sampled = dnsmpl_func(rang)

        return df_sampled
    
    def runtime_calc(self):
        """
        Function calculates the runtime for the simulation from the input data
        """
        
        #Extract max t_pressure time (in seconds)
        blad_dat = self.bladder_dat
        runtime = blad_dat["tpressure"].max()
        
        return runtime

    #Define function to smooth firing rate
    def fr_smooth(self, monitor, crit_freq, sampling_freq):
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

        #Import monitor and extract raw data
        dat = monitor.get_states()["rate"]

        #Define filter numerator and denomenator
        sos = butter(1, Wn=crit_freq, fs=sampling_freq, output="sos", analog=False)

        #Apply the filter to the data
        return sosfilt(sos, dat)

    def sim_build(self):
        """
        This method builds the framework for running a simulation, does so via:

        1. Defining neuronal/synaptic equations
        2. Imports and prepares bladder data (converts it to a brian readable format)
        3. Creates neuronal units from Brian2 NeuronGroup class
        4. Connects units via synapses, with weights specified by parameters supplied to class
        5. Specifies the duration for the simulation to be run

        Returns:
        ---------
        Monitors : list of objects
            The firing rate monitors that will collect neuronal information upon running the simulation

        Notes:
        ---------
        This method does not compile the code, instead it caches it temporarily.

        TO SAVE/COMPILE THIS NETWORK THE SIM_RUN() METHOD MUST BE CALLED. 
        IF A DIFFERENT BLADDERSIM OBJECT IS CREATED BEFORE THIS, THE DATA WILL BE OVERRIDDEN     
        """
        ###################################Experimental Parameters###################################
        
        #Build the device preferences, specifying any multicore processing required
        #Will run the simulation in standalone mode. 
        device.reinit()
        device.activate()

        set_device('cpp_standalone', build_on_run=False)
        prefs.devices.cpp_standalone.openmp_threads = self.cores #Num of cores to use in parallel.

        #Define default clock cycle (presently 50Hz, or 20ms)
        defaultclock.dt = 20*ms 
        
        #Specify seed for pseudorandom number generation (For testing only, disable this if running anlysis!)
        # device.seed(seed=4)

        #Specify experimental parameters
        update_time = 20 #The rate at which the timedarrays shall be updated

        #Tibial connectivity
        PMC_connected = True #Does the TN connect to PMC
        PAG_connected = True #Does the TN connect to the PAG
        Spine_connected = True #Does the TN connect to the spine? 

        #Tibial Strength. 
        tib_weight = [1.25, 0.45] #initial weight of tibial connections, fit from data. 
        op_w_max = 37.7 #Opioidergic weight fit from data. 
    
        tib_rate = self.tib_pars[0] #frequency of tibial input (Hz) 
        tib_dur = self.tib_pars[1] #duration of tibial input (s)

        ###################################Define Model Parameters###################################
        #Dynamics of a basic adaptive firing neuron

        #Membrane dynamics determined by Gorski et al. 
        Cm = 200*pF #Membrane capacitance, relatively similar across neurons
        E_A = -70*mV #Reversal potential of adaption conductance (outward current that drives hyperpolarisation)
        E_L = -60*mV #Resting potential (reversal potential of leak current)
        E_L_tonic = -70*mV 

        g_L = 10*nS #Resting conductance

        v_reset = -55*mV #Post-spike reset potential
        v_th = -50*mV #Spike threshold

        delta_t = 10*mV #Slope of the spike initiation (i.e, depolarisation) - PLACEHOLDER! 
        delta_t_ref = 5*ms #refractory period

        #Subthreshold adaption parameters
        v_A = -50*mV #Activation threshold for sub-spike adaption (drives changing firing rate)
        v_A_tonic = -45*mV 

        delta_A = 5*mV #Slope of subthreshold activation (severity of spike rate adaption)
        delta_g_A = 1*nS #Stepwise increase in adaption current (set to zero for non-adaption)
        delta_g_A_tonic = 0*nS

        g_A_max = 16.035434003920464*nS #Max adaptive current, change this?? 
        g_A_max_tonic = 2*nS 

        tau_A = 200*msecond #Decay rate for adaption current
        tau_A_tonic = 40*msecond 

        #Tonic firing parameter (causes self sustained firing)
        Iap = 111.63305739064285*pA #Reduce this, slow the rate of tonic independent? Its at like 20Hz rn...

        #Synaptic parameters
        E_ex = 0*mV #Reversal potential of excitatory current
        E_in = -80*mV #Reversal potential of inhibitory current

        #Opioidergic parameters (To be fit!)
        E_op = -80*mV #Reversal potential of enkephalinergic current 
        tau_op = 10*ms #Rate of enkephalinergic conductance decay, short term lingering inhibition. - To be fit
        
        g_op_increment = 6*nS #Baseline conductance of enkephalinergic synapse - to be fit
        
        ##Learning parameters, determined by Vogels et. al., 2011

        #First decay constants (For conductance)
        tau_glut = 5*ms #Excitatory rate of conductance decay (Glutamatergic transmission)
        tau_gaba = 10*ms #Inhibitory rate of conductance decay (GABAergic transmission)
        tau_stdp = 20*ms #Rate of memory trace decay

        #Then weight increments (To be fit)
        g_ex_increment = 10.046720218638525*nS #Baseline excitatory conductance
        g_in_increment = 14.901736779728395*nS #Inhibitory

        #Max weights
        ex_w_max = 34.35351387571368 #Max excitatory weight 
        in_w_max = 9.464573630030037 #Max inhibitory weigth 

        #Learning rates and depression parameters
        learn_rate = 5*10**-3 #Learning rate of synapse

        p_o = 1*Hz #Target postsynaptic firing rate

        dep_fac = 2*tau_stdp*p_o #Rate of synaptic depression

        #Population parameter
        pop = 100 #Can't be 1000, uses far too much memory. 

        ############################################################################################

        #Define Model Equations
        #First, set equation for tonically active (nonadaptive) neurons within the circuit
        #These neurons possess both an external current (Iap) and synaptic input (Is) for modulation
        #Iap is presumed to be constant

        tonic_eqs = """
        dv/dt = (g_L*(E_L_tonic - v) + g_L*delta_t*exp((v - v_th)/delta_t) + g_A*(E_A - v) + Iap + Is) / Cm : volt (unless refractory)
        dg_A/dt = ((g_A_max_tonic / (1 + exp((v_A_tonic - v)/delta_A))) - g_A) / tau_A_tonic : siemens
        Is = g_ex*(E_ex - v) + g_in*(E_in - v) + g_op*(E_op - v): amp
        dg_ex/dt = -g_ex / tau_glut : siemens
        dg_in/dt = -g_in / tau_gaba : siemens
        dg_op/dt = -g_op / tau_op : siemens
        """

        #Then, define equation for normal adaptive neurons
        #NOTE: the rate of adaption is determined by delta_g_A (the change in adaption with each spike)
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

        #Define effects of subthreshold adaption
        #Remember, delta_g_A determines rate of adaption
        on_spike = """
        v = v_reset 
        g_A += delta_g_A
        """

        on_spike_tonic = """
        v = v_reset 
        g_A += delta_g_A_tonic
        """

        #Define synaptic model to allow for learning
        #NOTE: a_pr & a_pst are synaptic traces (memory)
        #NOTE: traces updated only on spiking, to save memory
        syn_model = """
        weight : 1
        da_pr/dt = -a_pr/tau_stdp : 1 (event-driven)
        da_pst/dt = -a_pst/tau_stdp : 1 (event-driven)
        """

        #Define effects of presynaptic spiking
        #First for excitatory synapses
        on_pre_ex = """
        a_pr += 1 #Increase presynaptic memory trace
        weight = clip(weight + (learn_rate * (a_pst - dep_fac)), 0.01, ex_w_max) #Increment the weight based on postsynaptic activity
        g_ex += g_ex_increment * weight #Increase conductance representing transmission
        """

        #Then for inhibitory
        on_pre_in = """
        a_pr += 1 #Increase presynaptic memory trace
        weight = clip(weight + (learn_rate * (a_pst - dep_fac)), 0.01, in_w_max) #Increment the weight based on postsynaptic activity
        g_in += g_in_increment * weight #Increase conductance representing transmission
        """

        #Define effects of postsynaptic spiking
        #First for excitatory synapses
        on_post_ex = """
        a_pst += 1 #Increase postsynaptic memory trace
        weight = clip(weight + (learn_rate * a_pr), 0.01, ex_w_max) #Increment weight based on presynaptic activity
        """

        #Then inhibitory
        on_post_in = """
        a_pst += 1 #Increase postsynaptic memory trace
        weight = clip(weight + (learn_rate * a_pr), 0.01, in_w_max) #Increment weight based on presynaptic activity
        """

        #Finally, set up new equations for opioidergic transmission (inhibitory)
        on_pre_op="""
        a_pr += 1 #Increase memory trace (STDP)
        weight = clip(weight + (learn_rate * (a_pst - dep_fac)), 0, op_w_max)
        g_op += g_op_increment * weight
        """

        on_post_op="""
        a_pst += 1
        weight = clip(weight + (learn_rate * a_pr), 0, op_w_max)
        """

        ##################################BLADDER#################################################
        #First, led the bladder be defined as a single neuronal unit with no set equation
        #Instead with a series of dimensionless parameters:
        # Pressure (Pb)
        # Volume (V)
        
        #Import bladder data, keep at 50Hz (no sampling change)
        bladder_data = self.bladder_preprocess(50)

        #Extract specific bladder parameters
        volume_dat = bladder_data[0]
        pressure_dat = bladder_data[1]

        #-------------------Precalculate Afferent Parameters---------------------

        #Convert the finished data to a Brian2 readable array. 
        pressure_dat_array = TimedArray(pressure_dat, dt=update_time*ms)

        #Do the same for bladder volume
        volume_dat_array = TimedArray(volume_dat, dt=update_time*ms)

        #Now calculate pelvic and hypogastric afferent firing rates
        pel_rate=[]
        hyp_rate=[]

        #Enumerate over the pressure data
        for k, l in enumerate(pressure_dat):

            #Extract the currently recorded pressure
            current_pressure=pressure_dat[k]

            #From this, calculate the rate of the afferent projections
            #Using the relationship experimentally derived by McGee & Grill (2016)
            #Extrapolated to include hypogastric afferent projections in line with De Groat et al. 2013

            cal_rate = (-3*10**-8)*current_pressure**5 + (1*10**-5)*current_pressure**4 - (1.5*10**-3)*current_pressure**3 + (7.9*10**-2)*current_pressure**2 - 0.6*current_pressure
            
            #Append the real value if greater than the basal firing rate
            if cal_rate > 0:
                pel_rate.append(cal_rate)
                hyp_rate.append((-3*10**-8)*current_pressure**5 + (1*10**-5)*current_pressure**4 - (1.5*10**-3)*current_pressure**3 + (7.9*10**-2)*current_pressure**2 - 0.6*current_pressure)
                
            else:
                pel_rate.append(0)
                hyp_rate.append(0)
                

        #Convert the rates to Brian2 readable values
        pel_rate = TimedArray(pel_rate*Hz, dt=update_time*ms)
        hyp_rate = TimedArray(hyp_rate*Hz, dt=update_time*ms)

        #Calculate desired tibial nerve input
        #Create array of zeroes length of bladder data
        tib_dat = np.zeros(round(len(pressure_dat)))

        #Replace N values with rate
        num_cycles = int(tib_dur/(update_time/1000)) #convert from seconds to clock cycles (divide by s)
        tib_dat[0:num_cycles] = tib_rate
        
        #Convert to brian-readable timed array.
        in_input = TimedArray(tib_dat*Hz, dt=update_time*ms)

        #-----------------------------Define Bladder and Afferents------------------

        #First set the equation for the bladder as reading directly from these TimedArrays

        bladder_eqs = """
        Pb = pressure_dat_array(t) : 1
        V = volume_dat_array(t) : 1 
        """

        #Now define the detrusor
        #Using this equation
        detrusor = NeuronGroup(1, bladder_eqs)

        #Define the internal urethral sphincter
        ius = NeuronGroup(1, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        ius.v = "E_L + rand()*(v_th - E_L)"

        #Define the external urethral sphincter
        eus = NeuronGroup(1, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        eus.v = "E_L + rand()*(v_th - E_L)"

        ##################################Neuronal Group Definition#####################################

        #Tibial Nerve
        #-----------------------------------------------------------------------
        tibial_nerve = PoissonGroup(pop, rates="in_input(t)", name="TibialNerve") #TO BE UPDATED WITH RATES. 
        
        #Spinal Interneuron (GABAergic)
        tibial_classic = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        tibial_classic.v = "E_L + rand()*(v_th - E_L)"

        #Ascending Opioidergic
        tibial_op = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        tibial_op.v = "E_L + rand()*(v_th - E_L)"

        #Poisson Neurons
        #-----------------------------------------------------------------------
        #The Pelvic Nerve (NOTE: This is a poisson group whose firing rate is determined by Pb!)
        #Updates its firing rate with each timestep
        pelvic_aff = PoissonGroup(pop, rates="pel_rate(t)", name="PelvicAfferent")

        #The Hypogastric afferent
        hyp_aff = PoissonGroup(pop, rates="hyp_rate(t)", name="HypogastricAfferent")
        #-----------------------------------------------------------------------

        ##Brainstem Neurons##

        #First, Independent tonic neurons
        tonic_ind_pop = pop
        tonic_ind = NeuronGroup(pop, tonic_eqs, threshold="v>0*mV", reset=on_spike_tonic, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup")
        tonic_ind.v = "E_L + rand()*(v_th - E_L)"
        
        tonic_ind_PMC_pop = pop
        tonic_ind_PMC = NeuronGroup(pop, tonic_eqs, threshold="v>0*mV", reset=on_spike_tonic, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup2")
        tonic_ind_PMC.v = "E_L + rand()*(v_th - E_L)"
        
        #Then inverse neurons
        inv_neuron_pop = pop
        inv_neuron = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup3")
        inv_neuron.v = "E_L + rand()*(v_th - E_L)"
        
        #Type I direct neurons (RENAMED TO des_in from t1_direct)
        des_in_pop = pop
        des_in = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler", name = "TestGroup4")
        des_in.v = "E_L + rand()*(v_th - E_L)"
        
        #Type II direct neurons
        t2_direct_pop = pop
        t2_direct = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        t2_direct.v = "E_L + rand()*(v_th - E_L)"
        
        #Transient neurons
        transient_pop = pop
        transient = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        transient.v = "E_L + rand()*(v_th - E_L)"
        
        #Finally, Relay neurons
        #First pathway A (upper) relay
        relay_A_pop = pop
        relay_A = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        relay_A.v = "E_L + rand()*(v_th - E_L)"
        

        #Then pathway B (lower) relay
        relay_B_pop = pop
        relay_B = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        relay_B.v = "E_L + rand()*(v_th - E_L)"
        
        ##Pudendal Loop##
        #Pudendal motorneurons (Onuf's Nucleus)
        onuf = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        onuf.v = "E_L + rand()*(v_th - E_L)"

        #Preganglionic neurons within the spine
        pgn = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pgn.v = "E_L + rand()*(v_th - E_L)"

        #Pudendal afferent
        pud_aff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pud_aff.v = "E_L + rand()*(v_th - E_L)"

        #Pudendal Process
        pud_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pud_proc.v = "E_L + rand()*(v_th - E_L)"

        #Circuit connectivity
        pgn_stim = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pgn_stim.v = "E_L + rand()*(v_th - E_L)"

        #Then Circuit IO
        #Ascending output
        asc_out = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        asc_out.v = "E_L + rand()*(v_th - E_L)"

        #Onuf inhibition
        onuf_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        onuf_inhib.v = "E_L + rand()*(v_th - E_L)"

        ##Hypogastric loop##
        #First major hypogastric neuron groups
        hyp_stim = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        hyp_stim.v = "E_L + rand()*(v_th - E_L)"

        hyp_nerve = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        hyp_nerve.v = "E_L + rand()*(v_th - E_L)"

        img = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        img.v = "E_L + rand()*(v_th - E_L)"

        hyp_eff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        hyp_eff.v = "E_L + rand()*(v_th - E_L)"

        #Descending inhibition
        hyp_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        hyp_inhib.v = "E_L + rand()*(v_th - E_L)"

        ##Pelvic Loop##
        #The last reflex loop to be constructed is the connections within the pelvic nerve
        #First major neuron groups

        pelvic_eff = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pelvic_eff.v = "E_L + rand()*(v_th - E_L)"

        pelvic_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pelvic_proc.v = "E_L + rand()*(v_th - E_L)"

        #Including other connections
        hyp_proc = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        hyp_proc.v = "E_L + rand()*(v_th - E_L)"

        ##Remaining PGN Connections##
        #Inhibitory neuron driven by PGN
        pgn_inhib = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pgn_inhib.v = "E_L + rand()*(v_th - E_L)"

        #Inverse neuron that inhibits PGN via descending input
        des_inverse = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        des_inverse.v = "E_L + rand()*(v_th - E_L)"

        #PGN reciprocal inhibitor
        pgn_reciprocal = NeuronGroup(pop, adapt_eqs, threshold="v>0*mV", reset=on_spike, refractory=delta_t_ref, method="exponential_euler")
        pgn_reciprocal.v = "E_L + rand()*(v_th - E_L)"

        ##################################Synapses#################################################

        ##Tibial Nerve
        #----------------------------------------------------------------------

        #First from Tibial nerve to two interneurons (Excitatory, classical)
        tib_tib_classic = Synapses(tibial_nerve, tibial_classic, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
        tib_tib_classic.connect(condition='i!=j', p = 0.2)
        tib_tib_classic.weight = tib_weight[0]
        tib_clac = PopulationRateMonitor(tibial_classic)

        tib_tib_op = Synapses(tibial_nerve, tibial_op, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)
        tib_tib_op.connect(condition='i!=j', p = 0.2)
        tib_tib_op.weight = tib_weight[1]
        
        if Spine_connected is True:
            #Then from classical spinal branch to ascending output (Inhibitory, classical)
            tib_classic_asc_out = Synapses(tibial_classic, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)
            tib_classic_asc_out.connect(condition='i!=j', p = 0.2)
            tib_classic_asc_out.weight = tib_weight[0]

        if PMC_connected is True:
            #If this value is true then the tibial nerve will project to connections in the PMC
            #To transient neurons
            tib_op_transient = Synapses(tibial_op, transient, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_transient.connect(condition='i!=j', p = 0.2)
            tib_op_transient.weight = tib_weight[1]

            #To inverse neuron (inhibitory, enkephalinergic)
            tib_op_in = Synapses(tibial_op, inv_neuron, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_in.connect(condition='i!=j', p = 0.2)
            tib_op_in.weight = tib_weight[1]

            #To T2 direct (feedback)
            tib_op_t2 = Synapses(tibial_op, t2_direct, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_t2.connect(condition='i!=j', p = 0.2)
            tib_op_t2.weight = tib_weight[1]

            #To tonic independent
            tib_op_tonic_ind = Synapses(tibial_op, tonic_ind_PMC, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_tonic_ind.connect(condition='i!=j', p = 0.2)
            tib_op_tonic_ind.weight = tib_weight[1]

            #To type I direct (des input)
            tib_op_des_in = Synapses(tibial_op, des_in, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_des_in.connect(condition='i!=j', p = 0.2)
            tib_op_des_in.weight = tib_weight[1]
        
        if PAG_connected is True:
            #If this value is true, then the tibial nerve will project to the PAG of the brainstem
            #To the upper relay
            tib_op_relay_A = Synapses(tibial_op, relay_A, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_relay_A.connect(condition='i!=j', p = 0.2)
            tib_op_relay_A.weight = tib_weight[1]
            
            #To the lower relay
            tib_op_relay_B = Synapses(tibial_op, relay_B, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_relay_B.connect(condition='i!=j', p = 0.2)
            tib_op_relay_B.weight = tib_weight[1]

            #To the tonic independent neuron
            tib_op_tonic_ind_pag = Synapses(tibial_op, tonic_ind, model=syn_model, on_pre=on_pre_op, on_post=on_post_op)
            tib_op_tonic_ind_pag.connect(condition='i!=j', p = 0.2)
            tib_op_tonic_ind_pag.weight = tib_weight[1]
        
        #------------------------------------------------------------------------------       
        ###Brainstem Switch###
        ##PAG GROUPS##
        #First inhibitory connection from Independant -> Relay in PAG
        pag_I_R = Synapses(tonic_ind, relay_A, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)
        pag_I_R.connect(condition='i!=j', p = 0.2)
        pag_I_R.weight = self.params[0]

        #Then excitatory connection from Relay in PAG to Transient neuron & Type I direct
        #First to transient
        R_Tr = Synapses(relay_A, transient, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        R_Tr.connect(condition='i!=j', p = 0.2)
        R_Tr.weight = self.params[1],

        #From secondary relay (pathway B) to inverse neuron
        R2_In = Synapses(relay_B, inv_neuron, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        R2_In.connect(condition='i!=j', p = 0.2)
        R2_In.weight = self.params[2]

        #Then PAG relay to type I direct neuron
        R_Di = Synapses(relay_A, des_in, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        R_Di.connect(condition='i!=j', p = 0.2)
        R_Di.weight = self.params[3]

        #Then input from spine to relay 1 (excitatory)
        asc_out_R = Synapses(asc_out, relay_A, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        asc_out_R.connect(condition='i!=j', p = 0.2)
        asc_out_R.weight = self.params[4]

        #And input from spine to relay 2 (excitatory)
        asc_out_R2 = Synapses(asc_out, relay_B, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        asc_out_R2.connect(condition='i!=j', p = 0.2)
        asc_out_R2.weight = self.params[5]

        ##PMC GROUPS##
        #Then Transient to Inverse neuron
        Tr_In = Synapses(transient, inv_neuron, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        Tr_In.connect(condition='i!=j', p = 0.2)
        Tr_In.weight = self.params[6]

        #Then from type I direct to type II direct
        Di_Dii = Synapses(des_in, t2_direct, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        Di_Dii.connect(condition='i!=j', p = 0.2)
        Di_Dii.weight = self.params[7]

        #Fron type II direct to inverse neuron (Positive feedback)
        Dii_In = Synapses(t2_direct, inv_neuron, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        Dii_In.connect(condition='i!=j', p = 0.2)
        Dii_In.weight = self.params[8]

        #Inverse neuron to type I direct
        In_Di = Synapses(inv_neuron, des_in, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        In_Di.connect(condition='i!=j', p = 0.2)
        In_Di.weight = self.params[9]

        #Finally, the second tonically active independent neuron to type I direct
        I_Di = Synapses(tonic_ind_PMC, des_in, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        I_Di.connect(condition='i!=j', p = 0.2)
        I_Di.weight = self.params[10]

        ###Pudendal Loop##
        #First Onuf to EUS
        onuf_eus = Synapses(onuf, eus, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        onuf_eus.connect(condition='i!=j', p = 0.2)
        onuf_eus.weight = self.params[11]

        #EUS to pudendal afferent
        eus_pud_aff = Synapses(eus, pud_aff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        eus_pud_aff.connect(condition='i!=j', p = 0.2)
        eus_pud_aff.weight = self.params[12]

        #Pudendal afferent to pudendal process
        pud_aff_pud_proc = Synapses(pud_aff, pud_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pud_aff_pud_proc.connect(condition='i!=j', p = 0.2)
        pud_aff_pud_proc.weight = self.params[13]

        #Pudendal process to PGN stimulator (inhibitory)
        pud_proc_pgn_stim = Synapses(pud_proc, pgn_stim, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        pud_proc_pgn_stim.connect(condition='i!=j', p = 0.2)
        pud_proc_pgn_stim.weight = self.params[14]

        #Pudendal process to PGN
        pud_proc_pgn = Synapses(pud_proc, pgn, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        pud_proc_pgn.connect(condition='i!=j', p = 0.2)
        pud_proc_pgn.weight = self.params[15]

        #Pudendal process to ascending output gate
        pud_proc_asc_out = Synapses(pud_proc, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        pud_proc_asc_out.connect(condition='i!=j', p = 0.2)
        pud_proc_asc_out.weight = self.params[16]

        #PGN stimulator to PGN 
        pgn_stim_pgn = Synapses(pgn_stim, pgn, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pgn_stim_pgn.connect(condition='i!=j', p = 0.2)
        pgn_stim_pgn.weight = self.params[17]

        #Descending input to PGN (direct)
        des_in_pgn = Synapses(des_in, pgn, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        des_in_pgn.connect(condition='i!=j', p = 0.2)
        des_in_pgn.weight = self.params[18]

        #Descending input to PGN stimulator
        des_in_pgn_stim = Synapses(des_in, pgn_stim, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        des_in_pgn_stim.connect(condition='i!=j', p = 0.2)
        des_in_pgn_stim.weight = self.params[19]

        #Descending input to Onuf inhibiting interneuron
        des_in_onuf_inhib = Synapses(des_in, onuf_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        des_in_onuf_inhib.connect(condition='i!=j', p = 0.2)
        des_in_onuf_inhib.weight = self.params[20]

        #Onuf inhibitor to onuf's nucleus
        onuf_inhib_onuf = Synapses(onuf_inhib, onuf, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        onuf_inhib_onuf.connect(condition='i!=j', p = 0.2)
        onuf_inhib_onuf.weight = self.params[21]

        ##ypogastric Loop##
        #Hypogastric afferent to hypogastric stimulator (interneuron)
        hyp_aff_hyp_stim = Synapses(hyp_aff, hyp_stim, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        hyp_aff_hyp_stim.connect(condition='i!=j', p = 0.2)
        hyp_aff_hyp_stim.weight = self.params[22]

        #Hypogastric stimulator to hypogastric nerve (preganglionic)
        hyp_stim_hyp_nerve = Synapses(hyp_stim, hyp_nerve, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        hyp_stim_hyp_nerve.connect(condition='i!=j', p = 0.2)
        hyp_stim_hyp_nerve.weight = self.params[23]

        #Hypogastric nerve (preganglionic) to Inferior Mesenteric Ganglia
        hyp_nerve_img = Synapses(hyp_nerve, img, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        hyp_nerve_img.connect(condition='i!=j', p = 0.2)
        hyp_nerve_img.weight = self.params[24]

        #IMG to hypogastric efferent
        img_hyp_eff = Synapses(img, hyp_eff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        img_hyp_eff.connect(condition='i!=j', p = 0.2)
        img_hyp_eff.weight = self.params[25]

        #Now main hypogastric efferent connections
        hyp_eff_detrusor = Synapses(hyp_eff, detrusor, model="")
        hyp_eff_detrusor.connect()

        #Then to IUS (excitatory)
        hyp_eff_ius = Synapses(hyp_eff, ius, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        hyp_eff_ius.connect()
        hyp_eff_ius.weight = self.params[26]

        #---------------------------------------------------

        ##Then descending input to inhibitory interneuron
        des_in_hyp_inhib = Synapses(des_in, hyp_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        des_in_hyp_inhib.connect(condition='i!=j', p = 0.2)                                                    
        des_in_hyp_inhib.weight = self.params[27]

        #Finally, inhibitory interneuron to hypograstric nerve (preganglionic)
        hyp_inhib_hyp_nerve = Synapses(hyp_inhib, hyp_nerve, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        hyp_inhib_hyp_nerve.connect(condition='i!=j', p = 0.2)
        hyp_inhib_hyp_nerve.weight = self.params[28]

        #Pelvic afferent to ascending output
        pelvic_aff_asc_out = Synapses(pelvic_aff, asc_out, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pelvic_aff_asc_out.connect(condition='i!=j', p = 0.2)
        pelvic_aff_asc_out.weight = self.params[29]

        #Pelvic afferent to pelvic process
        pelvic_aff_pelvic_proc = Synapses(pelvic_aff, pelvic_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex) 

        pelvic_aff_pelvic_proc.connect(condition='i!=j', p = 0.2)
        pelvic_aff_pelvic_proc.weight = self.params[30]

        #Pelvic afferent to hypogastric process
        pelvic_aff_hyp_proc = Synapses(pelvic_aff, hyp_proc, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pelvic_aff_hyp_proc.connect(condition='i!=j', p = 0.2)
        pelvic_aff_hyp_proc.weight = self.params[31]

        #Pelvic process to Onuf's Nucleus
        pelvic_proc_onuf = Synapses(pelvic_proc, onuf, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pelvic_proc_onuf.connect(condition='i!=j', p = 0.2)
        pelvic_proc_onuf.weight = self.params[32]

        #Hypogastric process to hypogastric nerve
        hyp_proc_hyp_nerve = Synapses(hyp_proc, hyp_nerve, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        hyp_proc_hyp_nerve.connect(condition='i!=j', p = 0.2)
        hyp_proc_hyp_nerve.weight = self.params[33]

        #PGN to pelvic efferent
        pgn_pelvic_eff = Synapses(pgn, pelvic_eff, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pgn_pelvic_eff.connect(condition='i!=j', p = 0.2)
        pgn_pelvic_eff.weight = self.params[34]

        #Pelvic efferent to detrusor (not connected.)
        pelvic_eff_detrusor = Synapses(pelvic_eff, detrusor, model="")
        pelvic_eff_detrusor.connect()


        ##Remaining Circuitry##
        #Connect new neurons to circuit
        #First pgn to inhibitory interneuron (targeting ascending gate)
        pgn_pgn_inhib = Synapses(pgn, pgn_inhib, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pgn_pgn_inhib.connect(condition='i!=j', p = 0.2)
        pgn_pgn_inhib.weight = self.params[35]

        #Then pgn to its reciprocal neuron
        pgn_pgn_reciprocal = Synapses(pgn, pgn_reciprocal, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        pgn_pgn_reciprocal.connect(condition='i!=j', p = 0.2)
        pgn_pgn_reciprocal.weight = self.params[36]

        #pgn reciprocal to pgn stimulator (inhibitory)
        pgn_reciprocal_pgn_stim = Synapses(pgn_reciprocal, pgn_stim, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        pgn_reciprocal_pgn_stim.connect(condition='i!=j', p = 0.2)
        pgn_reciprocal_pgn_stim.weight = self.params[37]

        #Then inhibitory interneuron to ascending gate (inhibitory)
        pgn_inhib_asc_out = Synapses(pgn_inhib, asc_out, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        pgn_inhib_asc_out.connect(condition='i!=j', p = 0.2)
        pgn_inhib_asc_out.weight = self.params[38]

        #Descending input to inverse neuron
        des_in_des_inverse = Synapses(des_in, des_inverse, model=syn_model, on_pre=on_pre_ex, on_post=on_post_ex)

        des_in_des_inverse.connect(condition='i!=j', p = 0.2)
        des_in_des_inverse.weight = self.params[39]

        #Inverse neuron to pgn reciprocal (inhibitory)
        des_inverse_pgn_reciprocal = Synapses(des_inverse, pgn_reciprocal, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        des_inverse_pgn_reciprocal.connect(condition='i!=j', p = 0.2)
        des_inverse_pgn_reciprocal.weight = self.params[40]

        #Inverse neuron to inhibitory interneuron (inhibitory)
        des_inverse_pgn_inhib = Synapses(des_inverse, pgn_inhib, model=syn_model, on_pre=on_pre_in, on_post=on_post_in)

        des_inverse_pgn_inhib.connect(condition='i!=j', p = 0.2)
        des_inverse_pgn_inhib.weight = self.params[41]

        ##Define firing rate monitors, and specify their targets. 
        #Afferents
        pelvic_aff_mon = PopulationRateMonitor(pelvic_aff) #Input to circuit from bladder
        hyp_aff_mon = PopulationRateMonitor(hyp_aff)

        #Efferents
        pelvic_eff_mon = PopulationRateMonitor(pelvic_eff) #Pelvic efferent Activity
        hyp_eff_mon = PopulationRateMonitor(hyp_eff)
        pud_eff_mon = PopulationRateMonitor(onuf)
        asc_mon = PopulationRateMonitor(asc_out) #Ascending second order interneurons activity. 
    
        #Specify that when the simulation is built, it should run here. 
        print("Creating Simulation Network")
        run(self.runtime_calc()*second, report=self.report_func) #Note, will not run the sim until device.build called. 

        #Return state monitors to object
        self.monitors = [pressure_dat_array.values, 
            pelvic_aff_mon, 
            hyp_aff_mon,
            pelvic_eff_mon,
            hyp_eff_mon,
            pud_eff_mon,
            asc_mon,
            in_input]

    #Define function to take completed simulation and run it.
    def sim_run(self, saveloc="BladderSimulationCompiled"):
        """
        This method compiles the constructed neural network (sim_build()) as standalone C++ code, then runs it. 
        It will compile this network to the directory specified. If desired, this simulation may then be run independently, via main.cpp

        Arguments:
        ------------
        saveloc : path
            The directory to save the compiled simulation to. 
        
        NOTE: This method that a network be built in the first place (i.e., that you have supplied parameters, and bladder/neuronal data to sim_build)

        -----------------------------------------------------------------
        Returns the:

        Bladder Pressure
        Pelvic afferent firing rate
        Hypogastric afferent firing rate
        Pelvic efferent firing rate
        Hypogastric efferent firing rate
        Pudendal efferent firing rate
        Second order ascending afferent firing rate
        Timestamps for this neural data (if needed for optimisation)
        Firing rate of the tibial nerve
        ------------------------------------------------------------------

        """

        ##################################################################################
        #Run built simulation network according to supplied parameters.
        device.build(directory=saveloc, run=True, clean=True)

        #Extract data from monitors, smooth firing rates
        pressure = self.monitors[0]
        inp = self.fr_smooth(self.monitors[1], crit_freq=1, sampling_freq=50)
        hyp_inp = self.fr_smooth(self.monitors[2], crit_freq=1, sampling_freq=50)

        pel_out = self.fr_smooth(self.monitors[3], crit_freq=1, sampling_freq=50)
        hyp_out = self.fr_smooth(self.monitors[4], crit_freq=1, sampling_freq=50)
        pud_out = self.fr_smooth(self.monitors[5], crit_freq=1, sampling_freq=50)

        asc = self.fr_smooth(self.monitors[6], crit_freq=1, sampling_freq=50)
        asc_timestamps = self.monitors[6].t/ms 

        tib = self.monitors[7]

        ##################################################################################

        #Return smoothed firing rates, save as results. 
        self.results = [pressure, 
                inp, 
                hyp_inp,
                pel_out,
                hyp_out,
                pud_out,
                asc,
                asc_timestamps,
                tib]

    #Finally, define an additional method to plot the output of the class if required. 
    def sim_plot(self):
        """
        This method takes the results of a simulation (sim_run()), and plots the key inputs/outputs of the circuit.
        Returns a matplotlib figure object.
        """

        #Check if the simulation has been run before (i.e., if results have been generated)
        assert self.results is not None, "No sim results found. Did you run a simulation using sim_run()?"
        
        #Create subplots
        fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(3.54, 3.54), dpi=600)
        
        #First, pressure
        ax[0].plot(self.results[0], color="purple")

        #Wrap y axis label
        ax[0].set_ylabel("Pressure\n(cmH₂O)")

        #Then the afferents (Pelvic and Hypogastric)
        ax[1].plot(self.results[1]) #Pelvic
        ax[1].plot(self.results[2]) #Hypogastric
        ax[1].text(0, 18, "Afferent")
        
        ax[1].set_ylabel("")
        
        #Next up the efferents
        ax[2].plot(self.results[3]) #Pelvic
        ax[2].plot(self.results[4]) #Hypogastric
        ax[2].plot(self.results[5]) #Pudendal
        ax[2].text(0, 18, "Efferent")

        #Plot shared x-axis label
        ax[2].set_xlabel("Simulated Clock Cycles (20ms)")
        ax[2].set_ylabel("")
        
        #Plot shared y-axis for firing rate
        ax[2].text(-3500, 13, "Firing Rate (Hz)", rotation=90)
        
        #Define legend
        line = [
            Line2D([0], [0], color="blue", lw=4),
            Line2D([0], [0], color="orange", lw=4),
            Line2D([0], [0], color="green", lw=4)]
        
        fig.legend(line, ["Pelvic", "Hypogastric", "Pudendal"], bbox_to_anchor=(1.05, -0.01), title="", ncols=3)
        
        plt.close() #Stop fig from displaying twice. 
        return fig