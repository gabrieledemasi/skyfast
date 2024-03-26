##From a scratch of Stefano Rinaldi


from skyfast.coordinates import celestial_to_cartesian, cartesian_to_celestial, Jacobian, inv_Jacobian###da copiare

## Import general packages
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches, rcParams
import h5py
import warnings
from distutils.spawn import find_executable
from numba import njit, prange
from pathlib import Path
from tqdm import tqdm
import socket
from corner import corner
import dill
#import pyvo as vo


## Scipy
from scipy.stats import multivariate_normal as mn
from scipy.special import logsumexp


## Astropy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

import json
## Figaro
from figaro.mixture import DPGMM 
from figaro.credible_regions import ConfidenceArea, ConfidenceVolume, FindNearest_Volume, FindLevelForHeight
from figaro.transform import *
from figaro.utils import get_priors
from figaro.diagnostic import compute_entropy_single_draw, angular_coefficient
#from figaro.marginal import _marginalise
from figaro.load import save_density
#from figaro.load import load_single_event










class skyfast():

    """ Contains methods for the rapid localization of gravitational-wave hosts 
        based on FIGARO, an inference code that estimates multivariate 
        probability densities given samples from an unknown distribution 
        using a Dirichlet Process Gaussian Mixture Model (DPGMM).

    Args:
        max_dist:            Maximum distance (in Mpc) within which to search for the host
        cosmology            Cosmological parameters assumed for a flat ΛCDM cosmology
        glade_file           Path to the catalog file (.hdf5 file created with the create_glade pipeline)
        n_gal_to_plot        Number of galaxies to be plotted. If a catalog is built, this will be the number of galaxies in the catalog 
        true_host            Coordinates of the true host of the gravitational-wave event, if known. #GC: should we specify the dimensionality?
        host_name            Name of the host, if known.  #GC: should we include galaxy names in the catalog? 
        entropy              Boolean flag that determines whether to compute the entropy or not
        n_entropy_MC_draws   Number of Monte Carlo draws to compute the entropy of a single realisation of the DPGMM with the figaro.diagnostic function "compute_entropy_single_draw"
        entropy_step         Integer number indicating the frequency of entropy calculation, once every entropy_step samples are added
        entropy_ac_steps     Length (in steps) of the chunk of entropy data used to compute the angular coefficient
        n_sign_changes       Number of zero-crossings required to determine that the entropy has reached a plateau
        levels               Credible region levels 
        region_to_plot       Customizable region to plot #GC: but I don't get how it is menaged if it is outside the confidence levels (e.g. larger than 90)
        n_gridpoints:        Number of points in the 3D coordinate grid (ra, dec, dist)  
        virtual_observatory  Boolean flag indicating whether to plot in 2D
        labels               Plot labels #GC: al momento sono poi definiti a mano, possibile conflitto con i label di matplotlib, propongo di cambiare in plot_labels
        out_folder           Path to the output folder
        out_name             Name of the output of the current analysis #GC: should we upgrade this to a mandatory argument? 
        sampling time
        prior_pars           NIW prior parameters (k, L, nu, mu) for the mixture, typically inferred from the sample usinf the "get_prior" function in figaro.utils 
        alpha0               Initial guess for the concentration parameter of the DPGMM
        std                  Std parameter for the NIW prior
        incr_plot               
    """


    def __init__(self,
                    max_dist, 
                    cosmology           = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                    glade_file          = None,
                    n_gal_to_plot       = -1,
                    true_host           = None,
                    host_name           = 'Host',
                    entropy             = False,
                    n_entropy_MC_draws  = 1e4,
                    entropy_step        = 1,
                    entropy_ac_steps    = 500,
                    n_sign_changes      = 5,
                    levels              = [0.50, 0.90],
                    region_to_plot      = 0.9,
                    n_gridpoints        = [320, 180,360],
                    virtual_observatory = False,
                    labels              = ['$\\alpha \ \mathrm{[rad]}$', '$\\delta \ \mathrm{[rad]}$', '$D_{L} \ \mathrm{[Mpc]}$'],
                    out_folder          = './output',
                    out_name            = 'test', 
                    incr_plot           = False,
                    sampling_time       = False, 
                    prior_pars          = None,
                    alpha0              = 1,
                    std                 = 5
                    ):
        

        

        ## Gaussian Mixture
        self.log_dict = {}
        self.max_dist = max_dist
        self.bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        self.prior_pars = get_priors(bounds = self.bounds, std = std, probit = False )
        self.mix = DPGMM(self.bounds, prior_pars= self.prior_pars, alpha0 = alpha0, probit = False)

        '''
        prior_pars = [0.1, np.identity(3)*1e2, 4, np.array([ 0, 0, 0])]
        prior_pars = [0.01, np.array([[ 500,  0,  -0],
                                      [ 0,  100,   0],
                                      [-0,    0, 300]]), 30, np.array([ 0,  0, 0 ])]                                                                                          
        prior_pars = [0.01, np.array([[ 1,   0,  -0],
                                      [ 0, 100,   0],
                                      [-0,   0, 300]]), 10,  np.array([ 0,  0, 0 ])]                               
        '''
    


        ## Debug
        self.N_clu = []
        self.N_PT = []

        self.n_sign_changes = n_sign_changes

        ## Grid
        self.ra   = np.linspace(0,2*np.pi, n_gridpoints[0])[1:]
        self.dec  = np.linspace(-np.pi/2, np.pi/2., n_gridpoints[1])[1:-1]
        self.dist = np.linspace(0, max_dist, n_gridpoints[2])[1:]##remove points that cause measured 3d the be zero
        
        self.dD   = np.diff(self.dist)[0]
        self.dra  = np.diff(self.ra)[0]
        self.ddec = np.diff(self.dec)[0]
        
        # 3D grid
        grid = []
        measure_3d = []
        distance_measure_3d = []
        for ra_i in self.ra:
            for dec_i in self.dec:
                cosdec = np.cos(dec_i)
                for d_i in self.dist:
                    grid.append(np.array([ra_i, dec_i, d_i]))
                    measure_3d.append(cosdec*d_i**2)
                    distance_measure_3d.append(d_i**2)
        self.grid = np.array(grid)
        self.log_measure_3d = np.log(measure_3d).reshape(len(self.ra), len(self.dec), len(self.dist))
        self.distance_measure_3d = np.array(distance_measure_3d).reshape(len(self.ra), len(self.dec), len(self.dist))
        
        # 2D grid
        grid2d = []
        measure_2d = []
        for ra_i in self.ra:
            for dec_i in self.dec:
                grid2d.append(np.array([ra_i, dec_i]))
                measure_2d.append(np.cos(dec_i))
        self.grid2d = np.array(grid2d)
        self.log_measure_2d = np.log(measure_2d).reshape(len(self.ra), len(self.dec))
        
        # Meshgrid
        self.ra_2d, self.dec_2d = np.meshgrid(self.ra, self.dec)
        self.cartesian_grid = celestial_to_cartesian(self.grid)
        self.probit_grid = transform_to_probit(self.cartesian_grid, self.bounds)
        self.log_inv_J = -np.log(inv_Jacobian(self.grid)) - probit_logJ(self.probit_grid, self.bounds)
        self.inv_J = np.exp(self.log_inv_J)
        #self.inv_J = np.exp(-probit_logJ(self.probit_grid, self.bounds))

        self.volume_already_evaluated = False



        ## Credible regions levels
        self.levels      = np.array(levels)
        self.areas_N     = {cr:[] for cr in self.levels}
        self.volumes_N   = {cr:[] for cr in self.levels} 
        
        ## Sampling time 
        if sampling_time is None:
            self.sampling_time = ''
        
        ## Entropy
        self.entropy            = entropy #GC: self.entropy never used. Check
        self.entropy_step       = entropy_step
        self.entropy_ac_steps    = entropy_ac_steps
        self.N_for_ac           = np.arange(self.entropy_ac_steps)*self.entropy_step
        self.n_entropy_MC_draws = int(n_entropy_MC_draws)
        self.R_S                = []
        self.ac                 = []
        self.ac_cntr            = n_sign_changes
        self.i                  = 0
        self.flag_skymap        = False #GC: what is this? 
        if entropy == True:
            self.flag_skymap = False



        ## True host
        if true_host is not None:
            if len(true_host) == 2:
                self.true_host = np.concatenate((np.array(true_host), np.ones(1)))
            elif len(true_host) == 3:
                self.true_host = true_host
        else:
            self.true_host = true_host
        self.log_dict['true_host'] = true_host
        self.host_name = host_name
        if self.true_host is not None:
            self.pixel_idx  = FindNearest_Volume(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])

    
        
        ## Catalog
        self.catalog = None
        if  glade_file is not None:
            self.standard_cosmology = {'h': 0.674, 'om': 0.315, 'ol': 0.685}
            self.cosmology = cosmology
            self.cosmological_model = FlatLambdaCDM(H0=(self.cosmology['h']*100.) * u.km / u.s / u.Mpc, Om0=self.cosmology['om'])
            self.load_glade(glade_file)
            self.cartesian_catalog = celestial_to_cartesian(self.catalog)
            self.probit_catalog    = transform_to_probit(self.cartesian_catalog, self.bounds)
            self.log_inv_J_cat     = -np.log(inv_Jacobian(self.catalog)) - probit_logJ(self.probit_catalog, self.bounds)
            self.inv_J_cat         = np.exp(self.log_inv_J) #GC: never used in the code
        if n_gal_to_plot == -1 and self.catalog is not None:
            self.n_gal_to_plot = len(self.catalog)
        else:
            self.n_gal_to_plot = n_gal_to_plot
        if region_to_plot in self.levels:
            self.region = region_to_plot
        else:
            self.region = self.levels[0] #GC: What if it is larger than self.levels[1]? 
        self.virtual_observatory = virtual_observatory


        
        ## For loops #GC: non ho capito
        if incr_plot:
            self.next_plot = 20
        else:
            self.next_plot = np.inf

        


        ## Outputs
        self.out_name   = out_name
        self.labels     = labels
        self.out_folder = Path(out_folder).resolve()
        if not self.out_folder.exists():
            self.out_folder.mkdir()
        self.make_folders()
        if find_executable('latex'):
                rcParams["text.usetex"] = True




    def make_folders(self):
        """
        Makes folders for outputs
        """
        self.skymap_folder = Path(self.out_folder, 'skymaps')
        if not self.skymap_folder.exists():
            self.skymap_folder.mkdir(parents=True)
        
        self.log_folder = Path(self.out_folder, 'log')
        if not self.log_folder.exists():
            self.log_folder.mkdir(parents=True)    

        if self.catalog is not None:
            self.volume_folder = Path(self.out_folder, 'volume')
            if not self.volume_folder.exists():
                self.volume_folder.mkdir(parents=True)
            self.catalog_folder = Path(self.out_folder, 'catalogs')
            if not self.catalog_folder.exists():
                self.catalog_folder.mkdir(parents=True)
        if self.next_plot < np.inf:
            self.CR_folder = Path(self.out_folder, 'CR')
            if not self.CR_folder.exists():
                self.CR_folder.mkdir()
            self.gif_folder = Path(self.out_folder, 'gif')
            if not self.gif_folder.exists():
                self.gif_folder.mkdir()
        if self.entropy:
            self.entropy_folder = Path(self.out_folder, 'entropy')
            if not self.entropy_folder.exists():
                self.entropy_folder.mkdir()
        self.density_folder = Path(self.out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()


    

    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        self.means = []
        self.covs = []
        self.w = []
        self.log_w = []
        for comp, wi, logw in zip(self.mix.mixture, self.mix.w, self.mix.log_w):
            self.means.append(comp.mu)
            self.covs.append(comp.sigma)
            self.w.append(wi)
            self.log_w.append(logw)
        return np.sum(np.array([w*mn(mean, cov, allow_singular = True).pdf(x) for mean, cov, w in zip(self.means, self.covs, self.w)]), axis = 0)




    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        return logsumexp(np.array([w + mn(mean, cov, allow_singular = True).logpdf(x) for mean, cov, w in zip(self.means, self.covs, self.log_w)]), axis = 0)




    def load_glade(self, glade_file):
        """
        Load GLADE+ from hdf5 file.
        This is tailored to the GLADE+ hdf5 file created by the create_glade.py pipeline.
        
        Arguments:
            str or Path glade_file: glade file to be uploaded
        """
        self.glade_header =  ' '.join(['ra', 'dec', 'z', 'DL', 'm_B', 'm_K', 'm_W1', 'm_bJ', 'logp'])
        with h5py.File(glade_file, 'r') as f:
            dec = np.array(f['dec'])
            ra  = np.array(f['ra'])
            z   = np.array(f['z'])
            DL  = np.array(f['DL'])
            B   = np.array(f['m_B'])
            K   = np.array(f['m_K'])
            W1  = np.array(f['m_W1'])
            bJ  = np.array(f['m_bJ'])
        
        if self.cosmology!=self.standard_cosmology:
            DL = self.cosmological_model.luminosity_distance(z).value

        catalog = np.array([ra, dec, DL]).T
        
        self.catalog = catalog[catalog[:,2] < self.max_dist]
        catalog_with_mag = np.array([ra, dec, z, B, K, W1, bJ]).T
        self.catalog_with_mag = catalog_with_mag[catalog[:,2] < self.max_dist]

    


    def evaluate_skymap(self, final_map):
        """
        Marginalises volume map over luminosity distance to get the 2D skymap and compute credible areas
        
        Arguments:
            bool final_map: flag to raise if the inference is finished.
        """
        if not self.volume_already_evaluated or final_map:
            p_vol= self.density.pdf(celestial_to_cartesian(self.grid)) /inv_Jacobian(self.grid)
            #p_vol               = self._pdf_probit(self.probit_grid) /inv_Jacobian(self.grid)*self.inv_J
            #p_vol               = self._pdf_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol) 
            self.p_vol          = p_vol/self.norm_p_vol
            
            #print(self.p_vol, np.max(self.p_vol))
            #print('ev_sky_1')
            
            #By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    #print('err1')
                    self.log_p_vol = self.density._logpdf(celestial_to_cartesian(self.grid)) - np.log(inv_Jacobian(self.grid) ) - self.log_norm_p_vol
                    #self.log_p_vol = self._logpdf_probit(self.probit_grid) - np.log(inv_Jacobian(self.grid) )  + probit_logJ(self.probit_grid, self.bounds)- self.log_norm_p_vol
                    #self.log_p_vol = self._logpdf_probit(self.probit_grid)- np.log(inv_Jacobian(self.grid) ) - self.log_inv_J - self.log_norm_p_vol
                    #self.log_p_vol = self.density._logpdf(celestial_to_cartesian(self.grid)) + Jacobian(self.grid) - self.log_norm_p_vol
            #print('ev_sky_2')      
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True
            #print('ev_sky_3')
        self.p_skymap = (self.p_vol*self.dD*self.distance_measure_3d).sum(axis = -1)
        
        # By default computes log(p_skymap). If -infs are present, computes log_p_skymap
        with np.errstate(divide='raise'):
            try:
                self.log_p_skymap = np.log(self.p_skymap)
            except FloatingPointError:
                self.log_p_skymap = logsumexp(self.log_p_vol + np.log(self.dD) + np.log(self.distance_measure_3d), axis = -1)
        #print('ev_sky_4')
        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.log_p_skymap, self.ra, self.dec, log_measure = self.log_measure_2d, adLevels = self.levels)
        #print('ev_sky_5')
        for cr, area in zip(self.levels, self.areas):
            self.areas_N[cr].append(area)




    def make_skymap(self, sampling_time = None, final_map = True, ):
        """
        Produces a skymap.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        #print('make_sk_0')
        if sampling_time is not None:
            sampl_time_output = '_st_{sampling_time}_'
        else:
            sampl_time_output = ''

        self.evaluate_skymap(final_map)
        #print('make_sk_1')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 500, cmap = 'Reds')
        ax.set_rasterization_zorder(-10)
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'dashed')
        ax.clabel(c1, fmt = {l:'{0:.0f}\\%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='${0}'.format(self.mix.n_pts)+'\ \mathrm{samples}$', alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$')
        ax.set_ylabel('$\\delta \ \mathrm{[rad]}$')
        
        ax.legend(handles = handles, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)
        if final_map:
            fig.savefig(Path(self.skymap_folder, 'skymap_'+self.out_name+'_final.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, 'skymap_'+self.out_name+'_all.png'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, 'skymap_'+self.out_name+'_first_skymap.pdf'), bbox_inches = 'tight')
            if self.next_plot < np.inf:
                fig.savefig(Path(self.gif_folder, 'skymap_'+self.out_name+'_{}'.format(self.mix.n_pts)+'.png'), bbox_inches = 'tight')
      #  plt.show()
        plt.close()
        
        
        

    def marginal_prob(self, mix,  axis = -1):
        """
        Marginalises out one or more dimensions from a FIGARO mixture.
        
        Arguments:
            figaro.mixture.mixture draws: mixture
            int or list of int axis:      axis to marginalise on
        
        Returns:
            figaro.mixture.mixture: the marginalised mixture
        """
        # Circular import
        from figaro.mixture import mixture
        ax     = np.atleast_1d(axis)
        dim    = mix.dim - len(ax)
        
        means  = np.delete(mix.means, ax, axis = -1)
        covs   = np.delete(np.delete(mix.covs, ax, axis = -1), ax, axis = -2)
        bounds = np.delete(mix.bounds, ax, axis = 0)
        
        return mixture(means, covs, mix.w, bounds, dim,mix.n_cl, mix.n_pts, mix.alpha, probit = mix.probit)
    
    
    
    
    def evaluate_volume_map(self):
        """
        Evaluates volume map and compute credbile volumes
        """
        if not self.volume_already_evaluated:
            p_vol= self.mix.pdf(celestial_to_cartesian(self.grid)) /inv_Jacobian(self.grid)
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol) 
            self.p_vol          = p_vol/self.norm_p_vol
            
            #print(self.p_vol, np.max(self.p_vol), 'cia'), 
            #print('ev_sky_1')
            # By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                    print(self.log_p_vol)
                except FloatingPointError:
                    print('err1')
                    self.log_p_vol = self.mix._logpdf(celestial_to_cartesian(self.grid)) - np.log(inv_Jacobian(self.grid) ) - self.log_norm_p_vol
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True
            
        self.volumes, self.idx_CR, self.volume_heights = ConfidenceVolume(self.log_p_vol, self.ra, self.dec, self.dist, log_measure = self.log_measure_3d, adLevels = self.levels)
        # print('heights', self.log_p_vol, self.volume_heights)
        for cr, vol in zip(self.levels, self.volumes):
            self.volumes_N[cr].append(vol)




    def evaluate_catalog(self, final_map = False):
        """
        Evaluates the probability of being the host for each entry in the galaxy catalog and rank it accordingly.
        If the inference is finished, save credible areas/volumes.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        #log_p_cat = self.density._logpdf(self.cartesian_catalog) -inv_Jacobian(self.catalog)- self.log_norm_p_vol
        self.log_p_cat = self.density._logpdf(celestial_to_cartesian(self.catalog)) - np.log(inv_Jacobian(self.catalog) ) - self.log_norm_p_vol
        self.log_p_cat_to_plot     = self.log_p_cat[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.p_cat_to_plot         = np.exp(self.log_p_cat_to_plot)
        
        self.cat_to_plot_celestial = self.catalog[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.cat_to_plot_cartesian = self.cartesian_catalog[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        
        self.sorted_cat = np.c_[self.cat_to_plot_celestial[np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_cat_to_txt = np.c_[self.catalog_with_mag[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])][np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_p_cat_to_plot = np.sort(self.p_cat_to_plot)[::-1]
        
        np.savetxt(Path(self.catalog_folder, self.out_name+'_{0}'.format(self.mix.n_pts)+'.txt'), self.sorted_cat_to_txt, header = self.glade_header)
        if final_map:
            np.savetxt(Path(self.catalog_folder, 'CR_'+self.out_name+'.txt'), np.array([self.areas[np.where(self.levels == self.region)], self.volumes[np.where(self.levels == self.region)]]).T, header = 'area volume')

    
    
    
    def make_volume_map(self, final_map = False):
            """
            Produces self.catalogvolume map as 3D and 2D scatter plot of galaxies, if a catalog is provided.
            
            Arguments:
                bool final_map: flag to raise if the inference is finished
                int n_gals:     number of galaxies to plot
            """
            n_gals = self.n_gal_to_plot
            self.evaluate_volume_map()
            if self.catalog is None:
                return
            #print(self.catalog)
            self.evaluate_catalog(final_map)
            
            # Cartesian plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.cat_to_plot_cartesian[:,0], self.cat_to_plot_cartesian[:,1], self.cat_to_plot_cartesian[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.7, s = 0.5, cmap = 'Reds')
            vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
            vol_str = '\n'.join(vol_str + ['${0}'.format(len(self.cat_to_plot_cartesian)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'])
            ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            if final_map:
                fig.savefig(Path(self.volume_folder, self.out_name+'_cartesian_all.pdf'), bbox_inches = 'tight')
            else:
                fig.savefig(Path(self.volume_folder, self.out_name+'_cartesian_{0}'.format(self.mix.n_pts)+'.pdf'), bbox_inches = 'tight')
            plt.close()
            
            # Celestial plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.cat_to_plot_celestial[:,0], self.cat_to_plot_celestial[:,1], self.cat_to_plot_celestial[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.7, s = 0.5, cmap = 'Reds')
            vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
            vol_str = '\n'.join(vol_str + ['${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'])
            ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
            ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$')
            ax.set_ylabel('$\\delta \ \mathrm{[rad]}$')
            if final_map:
                fig.savefig(Path(self.volume_folder, self.out_name+'_final.pdf'), bbox_inches = 'tight')
                if self.next_plot < np.inf:
                    fig.savefig(Path(self.gif_folder, '3d_'+self.out_name+'_final.png'), bbox_inches = 'tight')
            else:
                fig.savefig(Path(self.volume_folder, self.out_name+'_first_skymap.pdf'), bbox_inches = 'tight')
                if self.next_plot < np.inf:
                    fig.savefig(Path(self.gif_folder, '3d_'+self.out_name+'_{0}'.format(self.mix.n_pts)+'.png'), bbox_inches = 'tight')
            #plt.show()
            plt.close()
            
            # 2D galaxy plot
            if self.virtual_observatory:
            # Limits for VO image
                fig_b = plt.figure()
                ax_b  = fig_b.add_subplot(111)
                c = ax_b.scatter(self.sorted_cat[:,0][:-int(n_gals):-1]*180./np.pi, self.sorted_cat[:,1][:-int(n_gals):-1]*180./np.pi, c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 1)
                x_lim = ax_b.get_xlim()
                y_lim = ax_b.get_ylim()
                plt.close(fig_b)
                fig = plt.figure()
                # Download background
                if self.true_host is not None:
                    pos = SkyCoord(self.true_host[0]*180./np.pi, self.true_host[1]*180./np.pi, unit = 'deg')
                else:
                    pos = SkyCoord((x_lim[1]+x_lim[0])/2., (y_lim[1]+y_lim[0])/2., unit = 'deg')
                size = (u.Quantity(4, unit = 'deg'), u.Quantity(6, unit = 'deg'))
                # To do: check this, pyvo has been commented
                ss = vo.regsearch(servicetype='image',waveband='optical', keywords=['SkyView'])[0]
                sia_results = ss.search(pos=pos, size=size, intersect='overlaps', format='image/fits')
                urls = [r.getdataurl() for r in sia_results]
                for attempt in range(10):
                    # Download timeout
                    try:
                        hdu = [fits.open(ff)[0] for ff in urls][0]
                    except socket.timeout:
                        continue
                    else:
                        break
                wcs = WCS(hdu.header)
                ax = fig.add_subplot(111, projection=wcs)
                ax.imshow(hdu.data,cmap = 'gray')
                ax.set_autoscale_on(False)
                c = ax.scatter(self.sorted_cat[:,0][:-int(n_gals):-1]*180./np.pi, self.sorted_cat[:,1][:-int(n_gals):-1]*180./np.pi, c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 0.5, transform=ax.get_transform('world'), zorder = 100)
                c1 = ax.contour(self.ra_2d*180./np.pi, self.dec_2d*180./np.pi, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'white', linewidths = 0.5, linestyles = 'solid', transform=ax.get_transform('world'), zorder = 99, alpha = 0)
                if self.true_host is not None:
                    ax.scatter([self.true_host[0]*180./np.pi], [self.true_host[1]*180./np.pi], s=80, facecolors='none', edgecolors='g', label = '$\mathrm{' + self.host_name + '}$', transform=ax.get_transform('world'), zorder = 101)
                leg_col = 'white'
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                #c = ax.scatter(self.sorted_cat[:,0][:-int(n_gals):-1], self.sorted_cat[:,1][:-int(n_gals):-1], c = self.sorted_p_cat_to_plot[:-int(n_gals):-1], marker = '+', cmap = 'coolwarm', linewidths = 1)
                c = ax.scatter(self.sorted_cat[:,0][:int(n_gals)], self.sorted_cat[:,1][:int(n_gals)], c = self.sorted_p_cat_to_plot[:int(n_gals)], marker = '+', cmap = 'coolwarm', linewidths = 1)
            

                #print('now', self.sorted_cat[:,0][:int(n_gals)])
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'solid')
                if self.true_host is not None:
                    ax.scatter([self.true_host[0]], [self.true_host[1]], s=80, facecolors='none', edgecolors='g', label = '$\mathrm{' + self.host_name + '}$')
                leg_col = 'black'
            for i in range(len(self.areas)):
                c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
            handles, labels = ax.get_legend_handles_labels()
            if self.n_gal_to_plot == -1 or self.n_gal_to_plot == len(self.catalog):
                lab_ngal = '${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'
            else:
                lab_ngal = '${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$\n'+'$({0}'.format(self.n_gal_to_plot)+'\ \mathrm{shown})$'
            patch = mpatches.Patch(color='grey', label=lab_ngal, alpha = 0)
            handles.append(patch)
            plt.colorbar(c, label = '$p_{host}$')
            ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$')
            ax.set_ylabel('$\\delta \ \mathrm{[rad]}$')
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.legend(handles = handles, loc = 2, fontsize = 10, handlelength=0, labelcolor = leg_col)
            if final_map:
                fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.out_name+'_final.pdf'), bbox_inches = 'tight')
            else:
                fig.savefig(Path(self.skymap_folder, 'galaxies_'+self.out_name+'_first_skymap.pdf'), bbox_inches = 'tight')
            #plt.show()    
            plt.close()




    def make_entropy_plot(self):
        """
        If entropy == True, produces entropy and angular coefficient plots.
        """
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.R_S))*self.entropy_step, (self.R_S), color = 'steelblue', lw = 0.7)
        ax.set_ylabel('$S(N)\ [\mathrm{bits}]$')
        ax.set_xlabel('$N$')
        
        fig.savefig(Path(self.entropy_folder, self.out_name + '.pdf'), bbox_inches = 'tight')
       #plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.axhline(0, lw = 0.5, ls = '--', color = 'r')
        ax.plot(np.arange(len(self.ac))*self.entropy_step + self.entropy_ac_steps, self.ac, color = 'steelblue', lw = 0.7)
        ax.set_ylabel('$\\frac{dS(N)}{dN}$')
        ax.set_xlabel('$N$')
        
        fig.savefig(Path(self.entropy_folder, 'ang_coeff_'+self.out_name + '.pdf'), bbox_inches = 'tight')
        #plt.show()
        plt.close()




    def density_from_samples(self, samples):
        """
        Produces a mixture from samples adding them one by one.

        Arguments:
            array samples: a (num,3) array containing num samples of (dl, ra, dec)
        """
        for s in tqdm(celestial_to_cartesian(samples)):
            self.mix.add_new_point(s)
        self.density = self.mix.build_mixture()
        #self.mix.initialise()
        #return ?? 
    



    def plot_samples(self, samples):
        """
        Draws samples from the inferred distribution and plots them.
        
        Arguments:
            array samples: a (num,3) array containing num samples of (dl, ra, dec)
        """
        samples_from_DPGMM = self.density.rvs(len(samples))
        c = corner(samples, color = 'black', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'})
        c = corner(cartesian_to_celestial(samples_from_DPGMM), fig = c,  color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}1$'})
        plt.legend(loc = 0,frameon = False,fontsize = 15)
        c.savefig(Path(self.skymap_folder, 'final.corner.png'))
        #plt.show()

    def save_density(self, final_map = False):
        """
        Build and save density
        """
        density = self.mix.build_mixture()
        if final_map == False:

            with open(Path(self.density_folder, self.out_name +f'first_skymap.pkl'), 'wb') as dill_file:
                dill.dump(density, dill_file)
        else:
            with open(Path(self.density_folder, self.out_name +f'_final.pkl'), 'wb') as dill_file:
                dill.dump(density, dill_file)

    
    def save_log(self):
        with open(Path(self.log_folder, self.out_name +f'log.json'), 'wb') as dill_file:
            json.dump(self.log_dict, dill_file)






        


    def intermediate_skymap(self, sample, sampling_time = None):
        """
        Adds a sample to the mixture, computes the entropy (if entropy == True), and releases an intermediate skymap as soon as convergence is reached.

        Arguments:
            3D array sample: one single sample (to be called in for loop giving samples one by one)
        """
        self.log_dict['sampling_time'] = sampling_time
        self.sampling_time = sampling_time
        self.mix.add_new_point(sample)
        self.density = self.mix.build_mixture()
        self.i +=1
        self.N_PT.append(self.mix.n_pts)
        self.N_clu.append(self.mix.n_cl)
        if self.entropy:
            if self.i%self.entropy_step == 0:
             
                R_S = compute_entropy_single_draw(self.density, self.n_entropy_MC_draws)
                self.R_S.append(R_S)
                if len(self.R_S)//self.entropy_ac_steps >= 1:
                    ac = angular_coefficient(np.array(self.N_for_ac + self.mix.n_pts), np.array(self.R_S[-self.entropy_ac_steps:]))
                    if self.flag_skymap == False:
                        try:
                            if ac*self.ac[-1] < 0:
                                self.ac_cntr = self.ac_cntr - 1
                        except IndexError: #Empty list
                            pass
                        if self.ac_cntr < 1:
                            self.log_dict['first_skymap_time'] = sampling_time
                            self.log_dict['first_skymap_samples'] = self.mix.n_pts
                            self.flag_skymap = True
                            #self.N.append(self.mix.n_pts)
                            self.make_skymap( sampling_time, final_map = False)
                            self.make_volume_map()
                            self.save_density()
                    self.ac.append(ac)


    def initialise(self): 
        """
        Initialises the existing instance of the skyfast class to new initial conditions. 
        This could be useful to analyze multiple GW events without the need of initializing skyfast from scratch (catalogue loading included) every time.
        """    

        self.mix.initialise()  

        self.R_S = []
        self.ac = []
        self.ac_cntr = self.n_sign_changes
        self.i = 0
               



        







        

if __name__ == "__main__":
    '''
    samples = np.genfromtxt('samples.dat', delimiter= ' ')
    #samples = np.genfromtxt()
    d = samples.T[0]
    ra = samples.T[1]
    dec = samples.T[2]

    samples = np.array([ra, dec, d]).T[1000:]
    c = corner(samples)
    plt.show()
    '''

    #samples, name = load_single_event('data/GW150914.hdf5', par = ['ra', 'dec', 'luminosity_distance'])

    samples, name = load_single_event('data/GW170817_noEM.txt')
    #samples, name = load_single_event('data/GW190814_posterior_samples.h5')
    glade_file = 'data/glade+.hdf5'
    ngc_4993_position = [3.446131245232759266e+00, -4.081248426799181650e-01]
    dens = skyfast(100, glade_file=glade_file,
                   true_host=ngc_4993_position,
                     entropy = True, 
                    n_entropy_MC_draws=1e3)#INSTANCE OF THE CLASS SKYFAST



    #samples = samples[::-1]
    '''
    samples = np.genfromtxt('samples.dat', delimiter= ' ')
    #samples = np.genfromtxt()
    d = samples.T[0]
    ra = samples.T[1]
    dec = samples.T[2]

    samples = np.array([dec, ra, d]).T[1000:]
    '''
    

    half_samples = samples
    
    cart_samp = celestial_to_cartesian(half_samples)
    np.random.shuffle(cart_samp)





    for i in tqdm(range(len(half_samples))):
        dens.intermediate_skymap(cart_samp[i])
    print('numero_cluster', dens.mix.n_cl)

    plt.figure(45)
    plt.plot(dens.N_PT, dens.N_clu)
    plt.figure(46)
    plt.plot(dens.N_PT, dens.R_S)
   #plt.show()

    dens.plot_samples(half_samples)
    dens.make_entropy_plot()

    dens.make_skymap(final_map = True)
    dens.make_volume_map(final_map = True)



