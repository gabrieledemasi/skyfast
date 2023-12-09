import numpy as np
from figaro.mixture import DPGMM
from figaro.load import load_single_event
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import multivariate_normal as mn
import matplotlib.patches as mpatches
samples, name = load_single_event('data/GW150914.hdf5', par = ['ra', 'dec', 'luminosity_distance'])
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from corner import corner
from figaro.coordinates import celestial_to_cartesian, cartesian_to_celestial, Jacobian, inv_Jacobian
from figaro.credible_regions import ConfidenceArea, ConfidenceVolume, FindNearest_Volume, FindLevelForHeight
from numba import njit, prange
from figaro.transform import *
from figaro.marginal import _marginalise
@njit
def log_add(x, y):
    """
    Compute log(np.exp(x) + np.exp(y))
    
    Arguments:
        double x: first addend (log)
        double y: second addend (log)
    
    Returns:
        double: log(np.exp(x) + np.exp(y))
    """
    if x >= y:
        return x+np.log1p(np.exp(y-x))
    else:
        return y+np.log1p(np.exp(x-y))

@njit
def log_add_array(x,y):
    """
    Compute log(np.exp(x) + np.exp(y)) element-wise
    
    Arguments:
        np.ndarray x: first addend (log)
        np.ndarray y: second addend (log)
    
    Returns:
        np.ndarray: log(np.exp(x) + np.exp(y)) element-wise
    """
    res = np.zeros(len(x), dtype = np.float64)
    for i in prange(len(x)):
        res[i] = log_add(x[i],y[i])
    return res

class skyfast():


    def __init__(self,
                    max_dist,
                    n_gridpoints = [360, 180, 50],
                    prior_pars = [0.01, np.array([[ 18.334944  ,  -1.43071565, -44.83109198],
                                                    [ -1.43071565, 278.20738326, 171.99388843],
                                                    [-44.83109198, 171.99388843, 363.21869174]]), 
                                                    100, 
                                                    np.array([ 117.05476008,  -11.53151026, -367.95616639])],
                    alpha0 = 1,
                    levels              = [0.50, 0.90],
                    out_folder          = '.',
                    latex               = False,
                    incr_plot           = False,
                    glade_file          = None,
                    cosmology           = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                    n_gal_to_plot       = -1,
                    region_to_plot      = 0.9,
                    entropy             = False,
                    n_entropy_MC_draws  = 1e3,
                    true_host           = None,
                    host_name           = 'Host',
                    entropy_step        = 1,
                    entropy_ac_step     = 500,
                    n_sign_changes      = 5,
                    virtual_observatory = False,
                    
                       
                       ):

        self.max_dist = max_dist
        self.bounds = np.array([[-max_dist, max_dist] for _ in range(3)])
        self.mix = DPGMM(self.bounds, prior_pars= prior_pars, alpha0 = 1, probit = False)
        self.levels =levels
        self.volume_already_evaluated = False
        self.latex = True

         # Grid
        self.ra   = np.linspace(0,2*np.pi, n_gridpoints[0])[1:]
        self.dec  = np.linspace(-np.pi/2, np.pi/2., n_gridpoints[1])[1:-1]
        self.dist = np.linspace(0, max_dist, n_gridpoints[2])[1:]##remove points that cause measured 3d the be zero
       
        self.dD   = np.diff(self.dist)[0]
        self.dra  = np.diff(self.ra)[0]
        self.ddec = np.diff(self.dec)[0]
        # For loops
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


        # Credible regions levels
        self.levels      = np.array(levels)
        self.areas_N     = {cr:[] for cr in self.levels}
        self.volumes_N   = {cr:[] for cr in self.levels}
        self.N           = []
        self.flag_skymap = False
        if entropy == True:
            self.flag_skymap = False


        #output
        self.out_folder = Path(out_folder).resolve()
        if not self.out_folder.exists():
            self.out_folder.mkdir()
        #self.make_folders()
    














    


    def evaluate_skymap(self):
        """
        Marginalise volume map over luminosity distance to get the 2D skymap and compute credible areas
        """
        if not self.volume_already_evaluated:
            p_vol= self.density.pdf(celestial_to_cartesian(self.grid)) * Jacobian(self.grid)
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol) 
            self.p_vol          = p_vol/self.norm_p_vol
            print('ev_sky_1')
            # By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                    print('err1')
                    self.log_p_vol = self.density._logpdf(celestial_to_cartesian(self.grid)) + Jacobian(self.grid) - self.log_norm_p_vol
            print('ev_sky_2')      
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True
            print('ev_sky_3')
        self.p_skymap = (self.p_vol*self.dD*self.distance_measure_3d).sum(axis = -1)
        
        # By default computes log(p_skymap). If -infs are present, computes log_p_skymap
        with np.errstate(divide='raise'):
            try:
                self.log_p_skymap = np.log(self.p_skymap)
            except FloatingPointError:
                self.log_p_skymap = logsumexp(self.log_p_vol + np.log(self.dD) + np.log(self.distance_measure_3d), axis = -1)
        print('ev_sky_4')
        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.p_skymap, self.ra, self.dec, log_measure = self.log_measure_2d, adLevels = self.levels)
        print('ev_sky_5')
        for cr, area in zip(self.levels, self.areas):
            self.areas_N[cr].append(area)

    def _pdf_probit(self, x):
        """
        Evaluate mixture at point(s) x in probit space.
        Overwrites parent method to avoid memory issues in 3D grid or catalog evaluation
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.pdf(x)
        """
        p = np.zeros(len(x))
        for comp, wi in zip(self.mix.mixture, self.mix.w):
            p += wi*mn(comp.mu, comp.sigma).pdf(x)
        return p
    
    def _logpdf_probit(self, x):
        """
        Evaluate log mixture at point(s) x in probit space.
        Overwrites parent method to avoid memory issues in 3D grid or catalog evaluation
        
        Arguments:
            np.ndarray x: point(s) to evaluate the mixture at (in probit space)
        
        Returns:
            np.ndarray: mixture.logpdf(x)
        """
        p = -np.ones(len(x))*np.inf
        for comp, wi in zip(self.mix.mixture, self.mix.log_w):
            p = log_add_array(p, wi + mn(comp.mu, comp.sigma).logpdf(x))
        return p  

    def make_skymap(self, final_map = False):
        """
        Produce skymap.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        print('make_sk_0')
        self.evaluate_skymap()
        print('make_sk_1')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 500, cmap = 'Reds')
        ax.set_rasterization_zorder(-10)
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'dashed')
        if self.latex:
            ax.clabel(c1, fmt = {l:'{0:.0f}\\%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        else:
            ax.clabel(c1, fmt = {l:'{0:.0f}%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='${0}'.format(self.density.n_pts)+'\ \mathrm{samples}$', alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha$')
        ax.set_ylabel('$\\delta$')
        ax.legend(handles = handles, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)

        
        

        plt.show()

    def marginal_prob(self, mix,  axis = -1):
        """
        Marginalise out one or more dimensions from a FIGARO mixture.
        
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



    def skymap_2d(self):
        p_vol = self.density.pdf(self.cartesian_grid).reshape(len(self.ra), len(self.dec), len(self.dist))[0]
        print()
        #norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
        plt.plot(p_vol.T)
        plt.show()





    


















    def build_density(self, samples):
        for s in tqdm(celestial_to_cartesian(samples)):
            self.mix.add_new_point(s)
        self.density = self.mix.build_mixture()
        self.mix.initialise()
        return 
    def plot_samples(self, samples):
        samples_from_DPGMM = self.density.rvs(8500)
        c = corner(samples, color = 'black', labels = ['$\\alpha$','$\\delta$', '$d$'], hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'}, range=([0, 2*np.pi], [-np.pi/2, np.pi/2], [0, 1000]))

        c = corner(cartesian_to_celestial(samples_from_DPGMM), fig = c,  color = 'dodgerblue', labels = ['$\\alpha$','$\\delta$', '$d$'], hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}1$'})
        l = plt.legend(loc = 0,frameon = False,fontsize = 15)
        plt.show()


    def plot_2d_contour(self):
        
        
        self.p_vol= self.density.pdf(celestial_to_cartesian(self.grid)) * Jacobian(self.grid)

        self.norm_p_vol     = (self.p_vol*np.exp(self.log_measure_3d.reshape(self.p_vol.shape))*self.dD*self.dra*self.ddec).sum()
        self.p_vol /= self.norm_p_vol
        self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))

        self.p_skymap = (self.p_vol*self.dD*self.distance_measure_3d).sum(axis = -1)
        fig, ax = plt.subplots()
        CS = ax.contour(self.ra_2d, self.dec_2d, self.p_skymap.T)
        ax.clabel(CS, inline=True, fontsize=10)
        ax.set_title('Simplest default with labels')
        plt.show()
        
        

    



dens = skyfast(1000)
dens.build_density(samples)
dens.plot_samples(samples)

dens.make_skymap(final_map = True)
#dens.plot_2d_contour()
#dens.skymap_2d()
