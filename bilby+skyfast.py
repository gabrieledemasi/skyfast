import bilby

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = "outdir11"
label = "fast_tutorial"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(88170235)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use two interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1). These default to their design
# sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
for key in [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    
    "geocent_time",
    "phase",
]:
    priors[key] = injection_parameters[key]



priors['chirp_mass'] = 28.28
priors['mass_ratio'] = 0.78


# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# Initialise the likelihood by passing in the interferometer data (ifos) and
# the waveform generator
likelihood = bilby.gw.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator
)


import numpy as np
from multiprocessing import Process
from skyfast import skyfast
from tqdm import tqdm
from figaro.coordinates import celestial_to_cartesian, cartesian_to_celestial, Jacobian, inv_Jacobian
import sys
import time 
rocket = 0
array_to_analize = np.array([])
def func1():
    result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="bilby_mcmc",
    nsamples=4000,
    npool = 6, 
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label)
    result.plot_corner()


def func2():
    glade_file = 'data/glade+.hdf5'
    dens = skyfast(4000,name ='injection',  glade_file=glade_file, n_gal_to_plot= 10, entropy = True, 
               n_entropy_MC_draws=1e3)#INSTANCE OF THE CLASS SKYFAST
    dens.ac_cntr = dens.n_sign_changes
    time.sleep(5)
    burn_in = 1000
    first_step = True
    step_analyze =20
    while(True):
        
        data = np.genfromtxt('samples.dat', delimiter= ' ')
        len_ = len(data)
        
        
        #print(len_)
        if len_>burn_in+50:
            if first_step:
                
                print(data[burn_in:burn_in +step_analyze], len(data[burn_in:burn_in +step_analyze]))#these are the samples that are added every 
                samples = data[burn_in:burn_in +step_analyze]
                d = samples.T[0]
                ra = samples.T[1]
                dec = samples.T[2]

                samples = np.array([dec, ra, d]).T


                cart_samp = celestial_to_cartesian(samples)
                np.random.shuffle(cart_samp)
                for i in tqdm(range(len(samples))):
                    dens.intermediate_skymap(cart_samp[i])
                
                start = burn_in + step_analyze
                print(start, 'dentro')
                first_step = False
            else:
                #len_ = len(data)
                if not len(data[start:start+ step_analyze])>0:
                    time.sleep(10)
                else:
                    samples = data[start:start+ step_analyze]
                    d = samples.T[0]
                    ra = samples.T[1]
                    dec = samples.T[2]

                    samples = np.array([dec, ra , d]).T


                    cart_samp = celestial_to_cartesian(samples)
                    np.random.shuffle(cart_samp)
                    for i in tqdm(range(len(samples))):
                        dens.intermediate_skymap(cart_samp[i])
                        print(data[start:start+ step_analyze], len(data[start:start+ step_analyze]))#these are the samples that are added every second
                    start = start+ step_analyze
                    print(start, 'fuori')   

        

        
       

                


        




p1 = Process(target=func1)
p1.start()
p2 = Process(target=func2)
p2.start()




