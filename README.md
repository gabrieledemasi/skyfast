# SKYFAST: Rapid localization of gravitational wave hosts

SKYFAST is a pipeline designed for the rapid localization of gravitational wave hosts based on Bayesian non-parametrics. It operates alongside a full parameter estimation (PE) algorithm, from which posterior samples are taken. These samples are then used to reconstruct an analytical posterior for the sky position, luminosity distance, and inclination angle using a Dirichlet Process Gaussian Mixture Model (DPGMM). Specifically, SKYFAST is based on FIGARO, an inference code designed to estimate multivariate probability densities given samples from an unknown distribution using a DPGMM as a non-parameteric model ([Rinaldi & del Pozzo 2024](https://joss.theoj.org/papers/10.21105/joss.06589)). FIGARO is publicly available [here](https://github.com/sterinaldi/FIGARO). 

The DPGMM approach allows for accurate localization of the event using only a fraction of the total samples produced by the PE run. Depending on the PE algorithm employed, this can lead to significant time savings, which is crucial for identifying the electromagnetic counterpart.

Within a few minutes, SKYFAST also generates a ranked list of the most probable galaxy hosts from the GLADE+ catalog. This list includes information on the inclination angle posterior conditioned to the position of each candidate host, which is useful for assessing the detectability of gamma-ray burst (GRB) structured jet emissions.

## Installation

We recomment to create a conda environment with Python 3.11:

`conda create --name skyfast_env python==3.11`.


Then activate it:

`conda activate skyfast_env`


Lastly, download SKYFAST from git and install it:

`git clone git@github.com:gabrieledemasi/skyfast.git`

`cd skyfast`

`pip install .`


## Getting started

An introductory guide, where we show how to use SKYFAST to reconstruct the posterior from the PE samples, produce a skymap of the GW event, and generate a list of the most probable galaxy hosts from the GLADE+ catalog, can be found [here](https://github.com/gabrieledemasi/skyfast/blob/main/Tutorial/tutorial_SKYFAST.ipynb)



## Acknowledgements

If you use SKYFAST in your research, please cite Demasi et al. 2024, in preparation.
