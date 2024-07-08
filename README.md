# SKYFAST: Rapid localization of gravitational wave hosts

SKYFAST is a pipeline designed for the rapid localization of gravitational wave hosts based on Bayesian non-parametrics. It operates alongside a full parameter estimation (PE) algorithm, from which posterior samples are taken. These samples are then used to reconstruct an analytical posterior for the sky position, luminosity distance, and inclination angle using a Dirichlet Process Gaussian Mixture Model (DPGMM). Specifically, we use the DPGMM implementation FIGARO ([Rinaldi & del Pozzo 2024](https://joss.theoj.org/papers/10.21105/joss.06589)), publicly available [here](https://github.com/sterinaldi/FIGARO). 

This approach allows for accurate localization of the event using only a fraction of the total samples produced by the PE run. Depending on the PE algorithm employed, this can lead to significant time savings, which is crucial for identifying the electromagnetic counterpart.

Within a few minutes, \texttt{SKYFAST} also generates a ranked list of the most probable galaxy hosts from the GLADE+ catalog. This list includes information on the inclination angle posterior conditioned to the position of each candidate host, which is useful for assessing the detectability of gamma-ray burst (GRB) structured jet emissions.

## Getting started

Create a conda environment:

`conda create --name skyfast_env python==3.11`.

Then activate it:
`conda activate skyfast_env`

Then download the package from git:

`git clone git@github.com:gabrieledemasi/skyfast.git`

Move in the downloaded directory and install the SKYFAST package:

`python setup.py install`

## Acknowledgements

If you use SKYFAST in your research, please cite Demasi et al. 2024, in preparation.
