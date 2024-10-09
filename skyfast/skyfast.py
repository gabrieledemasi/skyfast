##From a scratch of Stefano Rinaldi


## Import general packages
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches, rcParams
from matplotlib.ticker import ScalarFormatter
import h5py
from distutils.spawn import find_executable
from pathlib import Path
from corner import corner
import json
from figaro.load import save_density
from figaro.marginal import condition


## Astropy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


## Figaro
from figaro.mixture import DPGMM 
from figaro.credible_regions import ConfidenceArea, ConfidenceVolume, FindNearest_Volume
from figaro.transform import *
from figaro.utils import get_priors
from figaro.diagnostic import compute_entropy_single_draw, angular_coefficient
from figaro.marginal import marginalise
from figaro.utils import rvs_median

#from skyfast.coordinates import celestial_to_cartesian, cartesian_to_celestial, Jacobian, inv_Jacobian




class skyfast():

    """ Class that contains methods for the rapid localization of GW hosts 
        based on FIGARO, an inference code that estimates multivariate 
        probability densities given samples from an unknown distribution 
        using a Dirichlet Process Gaussian Mixture Model (DPGMM).

    Args:
        max_dist:            Maximum distance (in Mpc) within which to search for the host
        cosmology            Cosmological parameters assumed for a flat ΛCDM cosmology
        glade_file           Path to the catalog file (.hdf5 file created with the create_glade pipeline)
        n_gal_to_plot        Maximum number of galaxies in the 90% CR to be plotted in the skymaps
        true_host            Coordinates of the true host of the GW event, if known. 
        host_name            Name of the host, if known. 
        entropy              Boolean flag that determines whether to apply the information entropy convergence criterion
        n_entropy_MC_draws   Number of Monte Carlo draws to compute the entropy of a single realisation of the DPGMM with the figaro.diagnostic function "compute_entropy_single_draw"
        entropy_step         Integer number indicating how often the entropy is calculated (once every "entropy_step samples" are added)
        entropy_ac_steps     Length (in steps) of the chunk of entropy data used to compute the angular coefficient
        n_sign_changes       Number of zero-crossings required to determine if the entropy has reached a plateau
        levels               Credible region levels 
        region_to_plot       Customizable region to plot 
        n_gridpoints:        Number of points in the 3D coordinate grid (ra, dec, dL)  
        out_folder           Path to the output folder
        out_name             Name of the output from the current analysis 
        sampling time        Boolean flag that determines whether the funcion "intermediate_skymap" acquires the sampling time of the PE pipeline 
        prior_pars           NIW prior parameters (k, L, nu, mu) for the gaussian mixture, typically inferred from the samplez using the "get_prior" function from figaro.utils 
        alpha0               Initial guess for the concentration parameter of the DPGMM
        inclination          Boolean flag that determines wether the inclination angle is included in the analysis   
        true_inclination     True inclination of the GW event, if known.
        theta_condition      Boolean flag that determines wether to compute the inclination angle posterior conditioned to the position of each galaxy in the list 
        max_n_gal_cond       Maximum number of galaxies in the 90% CR for which to compute the conditioned inclination angle posterior   
    """


    def __init__(self,
                    max_dist            = 5000, 
                    cosmology           = {'h': 0.674, 'om': 0.315, 'ol': 0.685},
                    glade_file          = None,
                    n_gal_to_plot       = -1,
                    true_host           = None,
                    host_name           = 'Host',
                    entropy             = False,
                    n_entropy_MC_draws  = 1e3,
                    entropy_step        = 1,
                    entropy_ac_steps    = 200,
                    n_sign_changes      = 3,
                    levels              = [0.50, 0.90],
                    region_to_plot      = 0.9,
                    n_gridpoints        = [320, 180, 360],
                    out_folder          = './output',
                    out_name            = 'test', 
                    sampling_time       = False, 
                    prior_pars          = None,
                    alpha0              = 1, 
                    inclination         = False,
                    true_inclination    = None,
                    theta_condition     = False,
                    max_n_gal_cond      = None,
                    ):
        


        
        self.log_dict = {}  
        self.max_dist = max_dist
        self.true_host = true_host
        self.samples  = []
        eps = 1e-3

        self.inclination = inclination
        self.theta_condition = theta_condition
        self.max_n_gal_cond  = max_n_gal_cond
        if self.inclination ==False:
            self.bounds = np.array([[0.-eps, 2*np.pi+eps], [-np.pi/2 -eps, np.pi/2+eps], [0.-eps, self.max_dist+eps]])
        else:
            self.bounds = np.array([[0.-eps, 2*np.pi+eps], [-np.pi/2 -eps, np.pi/2+eps], [0.-eps, max_dist+eps], [0, np.pi]])

        if prior_pars is not None:
            self.prior_pars = prior_pars

        self.mix = DPGMM(self.bounds, prior_pars= self.prior_pars, alpha0 = alpha0, probit = True)

 
        ## Debug
        self.N_clu = []
        self.N_PT = []
        self.n_sign_changes = n_sign_changes

        ## Grid
        self.ra   = np.linspace(0,2*np.pi, n_gridpoints[0])[1:-1]
        self.dec  = np.linspace(-np.pi/2, np.pi/2., n_gridpoints[1])[1:-1]
        self.dist = np.linspace(0, max_dist, n_gridpoints[2])[1:]##remove points that cause measured 3d the be zero
        
        self.dD   = np.diff(self.dist)[0]
        self.dra  = np.diff(self.ra)[0]
        self.ddec = np.diff(self.dec)[0]
        
        # 3D grid
        grid = []
        measure_3d = []
        distance_measure_3d = []
      

        ###GRID###
        d2 = np.transpose([np.repeat(self.ra, len(self.dec)), np.tile(self.dec, len(self.ra))])
        self.grid = np.column_stack((np.repeat(d2, len(self.dist), axis = 0),np.tile(self.dist, len(d2))))
        
        #measure3d
        output1 = np.dstack(np.outer(self.dist**2, np.cos(self.dec))).reshape( -1,2) 

        measure_3d = np.tile(output1.T,len(self.ra) ).T


    
        output1 = np.tile((self.dist**2).T,len(self.dec)).T
        distance_measure_3d= np.tile(output1 .T,len(self.ra) ).T

        self.grid2d = d2

        #self.grid = np.array(grid)
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

        measure_2d = np.tile(np.cos(self.dec), len(self.ra))
        self.log_measure_2d = np.log(measure_2d).reshape(len(self.ra), len(self.dec))

        # Meshgrid
        self.ra_2d, self.dec_2d = np.meshgrid(self.ra, self.dec)
        #self.cartesian_grid = celestial_to_cartesian(self.grid)
        self.probit_grid = transform_to_probit(self.grid, self.bounds[:3])
        self.log_inv_J =  - probit_logJ(self.probit_grid, self.bounds[:3])
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
        self.flag_skymap        = False  
        if entropy == True:
            self.flag_skymap = False



        ## True host
        if true_host is not None:
            if len(true_host) == 2:
                self.true_host = np.concatenate((np.array(true_host), np.ones(1)))
            elif len(true_host) == 3:
                self.true_host = true_host 
            self.log_dict['true_host'] = list(true_host)
        else:
            self.true_host = true_host
  
        self.host_name = host_name
        if self.true_host is not None:
            self.pixel_idx  = FindNearest_Volume(self.ra, self.dec, self.dist, self.true_host)
            self.true_pixel = np.array([self.ra[self.pixel_idx[0]], self.dec[self.pixel_idx[1]], self.dist[self.pixel_idx[2]]])

        self.true_inclination = true_inclination

       
        
        ## Catalog
        self.catalog = None
        if  glade_file is not None:
            self.standard_cosmology = {'h': 0.674, 'om': 0.315, 'ol': 0.685}
            self.cosmology = cosmology
            self.cosmological_model = FlatLambdaCDM(H0=(self.cosmology['h']*100.) * u.km / u.s / u.Mpc, Om0=self.cosmology['om'])
            self.load_glade(glade_file)
            #self.cartesian_catalog = celestial_to_cartesian(self.catalog)
            self.probit_catalog    = transform_to_probit(self.catalog, self.bounds[:3])
            self.log_inv_J_cat     =  - probit_logJ(self.probit_catalog, self.bounds[:3])
            self.inv_J_cat         = np.exp(self.log_inv_J_cat) 
        if n_gal_to_plot == -1 and self.catalog is not None:
            self.n_gal_to_plot = len(self.catalog)
        else:
            self.n_gal_to_plot = n_gal_to_plot
        if region_to_plot in self.levels:
            self.region = region_to_plot
        else:
            self.region = self.levels[0] #GC: What if it is larger than self.levels[1]? 




        


        ## Outputs
        self.out_name   = out_name
            #labels
        if self.inclination:
            self.labels = ['$\\alpha \ \mathrm{[rad]}$', '$\\delta \ \mathrm{[rad]}$', '$d_{L} \ \mathrm{[Mpc]}$','$\\theta_{jn} \ \mathrm{[rad]}$' ]
        else:
            self.labels = ['$\\alpha \ \mathrm{[rad]}$', '$\\delta \ \mathrm{[rad]}$', '$d_{L} \ \mathrm{[Mpc]}$']
        ## Gaussian Mixture Parameters and initialization
        self.out_folder = Path(out_folder).resolve()
        if not self.out_folder.exists():
            self.out_folder.mkdir()
        self.make_folders()
        if find_executable('latex'):
                rcParams["text.usetex"] = True




    def make_folders(self):
        """
        Make folders for outputs
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
            self.hosts_folder = Path(self.out_folder, 'hosts')
            if not self.hosts_folder.exists():
                self.hosts_folder.mkdir(parents=True)
        
        self.entropy_folder = Path(self.out_folder, 'entropy')
        if not self.entropy_folder.exists():
            self.entropy_folder.mkdir()
        
        self.density_folder = Path(self.out_folder, 'density')
        if not self.density_folder.exists():
            self.density_folder.mkdir()

        self.corner_folder = Path(self.out_folder, 'corner')
        if not self.corner_folder.exists():
            self.corner_folder.mkdir() 

        self.CR_folder = Path(self.out_folder, 'CR')
        if not self.CR_folder.exists():
            self.CR_folder.mkdir()   
        

        


    def load_glade(self, glade_file):
        """
        Load GLADE+ catalog from a .hdf5 file.
        This is tailored to the glade.hdf5 file created by the create_glade.py pipeline.
        
        Arguments:
            str or Path glade_file: glade file to be uploaded
        """
        self.glade_header =  ' '.join(['glade_no', 'ra', 'dec', 'dL', 'm_B', 'm_K', 'm_W1', 'm_bJ', 'logp'])
        self.glade_header_cond = ' '.join(['glade_no','ra', 'dec', 'dL', 'ddL','m_B', 'm_K', 'm_W1', 'm_bJ', 'logp', 'theta_jn', 'delta_theta_plus (90% CR)', 'delta_theta_minus (90% CR)' ])
        with h5py.File(glade_file, 'r') as f:
            glade_no = np.array(f['glade_no'])
            dec = np.array(f['dec'])
            ra  = np.array(f['ra'])
            z   = np.array(f['z'])
            dL  = np.array(f['dL'])
            ddL = np.array(f['ddL'])
            B   = np.array(f['m_B'])
            K   = np.array(f['m_K'])
            W1  = np.array(f['m_W1'])
            bJ  = np.array(f['m_bJ'])
        
        if self.cosmology!=self.standard_cosmology:
            dL = self.cosmological_model.luminosity_distance(z).value
        #dL = self.cosmological_model.luminosity_distance(z).value
        catalog = np.array([ra, dec, dL]).T
        
        self.catalog = catalog[catalog[:,2] < self.max_dist]
        catalog_with_mag = np.array([glade_no, ra, dec, dL, ddL,  B, K, W1, bJ]).T
        self.catalog_with_mag = catalog_with_mag[catalog[:,2] < self.max_dist]

    

    
    def evaluate_skymap(self, final_map):
        """
        Marginalise volume map over luminosity distance to get the 2D skymap and compute credible areas
        
        Arguments:
            bool final_map: flag to raise if the inference is finished.
        """
        if not self.volume_already_evaluated or final_map:
            if self.inclination==False:
                self.vol_density = self.density
                self.map_density = marginalise(self.density, [2])
            else:
                self.vol_density = marginalise(self.density, [3])
                self.map_density = marginalise(self.density, [2,3])
                self.incl_density = marginalise(self.density, [0, 1, 2])


            self.p_vol = self.vol_density.pdf(self.grid)
            self.norm_p_vol     = np.sum(self.p_vol*np.exp(self.log_measure_3d.reshape(self.p_vol.shape))*self.dD*self.dra*self.ddec)

            with np.errstate(divide='ignore'):
                self.log_p_vol = np.log(self.p_vol)

    
            self.log_norm_p_vol = np.log(self.norm_p_vol) 
            self.p_vol          = self.p_vol/self.norm_p_vol
            self.log_p_vol     -= self.log_norm_p_vol

            
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True

        self.p_skymap =  self.map_density.pdf(self.grid2d)
        self.norm_skymap = np.sum(self.p_skymap*np.exp(self.log_measure_2d.reshape(self.p_skymap.shape))*self.dra*self.ddec)
        self.p_skymap/= self.norm_skymap
       
   
        #self.p_skymap = (self.p_vol*self.dD*self.distance_measure_3d).sum(axis = -1)
        with np.errstate(divide='ignore'):
            self.log_p_skymap = np.log(self.p_skymap) 
       
        self.log_p_skymap = self.log_p_skymap.reshape(len(self.ra), len(self.dec))
        self.p_skymap = self.p_skymap.reshape(len(self.ra), len(self.dec))
        # By default computes log(p_skymap). If -infs are present, computes log_p_skymap
        '''
        with np.errstate(divide='raise'):
            try:
                self.log_p_skymap = np.log(self.p_skymap)
            except FloatingPointError:
                self.log_p_skymap = logsumexp(self.log_p_vol + np.log(self.dD) + np.log(self.distance_measure_3d), axis = -1)
        '''
        self.areas, self.skymap_idx_CR, self.skymap_heights = ConfidenceArea(self.log_p_skymap, self.ra, self.dec, log_measure = self.log_measure_2d, adLevels = self.levels)
        for cr, area in zip(self.levels, self.areas):
            self.areas_N[cr].append(area)




    def make_skymap(self, final_map, show = False):
        """
        Produce the skymap of the GW event, with 50% and 90% CR.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """

        self.evaluate_skymap(final_map)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Set limits for the axes
        
        max_level_idx = self.skymap_idx_CR[len(self.levels) - 1] 
        ra_min = np.min(self.ra[max_level_idx.T[0]])
        ra_max = np.max(self.ra[max_level_idx.T[0]])
        delta_ra = ra_max - ra_min
        dec_min = np.min(self.dec[max_level_idx.T[1]])
        dec_max = np.max(self.dec[max_level_idx.T[1]])
        delta_dec = dec_max - dec_min
        x_lim = [max(ra_min - delta_ra,0.), min(ra_max + delta_ra, 2*np.pi)] 
        y_lim = [max(dec_min - delta_dec, -np.pi/2), min(dec_max + delta_dec, np.pi/2)]  
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
       
        # Plot skymap
        c = ax.contourf(self.ra_2d, self.dec_2d, self.p_skymap.T, 500, cmap = 'Reds')
       
        # Plot contours
        ax.set_rasterization_zorder(-10)
        c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'dashed')
        ax.clabel(c1, fmt = {l:'{0:.0f}%'.format(100*s) for l,s in zip(c1.levels, self.levels[::-1])}, fontsize = 5)
        
        # Legend and labels
        for i in range(len(self.areas)):
            c1.collections[i].set_label('${0:.0f}\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.1f}'.format(self.areas[-i]) + '\ \mathrm{deg}^2$')
        handles, labels = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='grey', label='${0}'.format(self.mix.n_pts)+'\ \mathrm{samples}$', alpha = 0)
        handles.append(patch)
        ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$')
        ax.set_ylabel('$\\delta \ \mathrm{[rad]}$')
        try:
            ax.legend(handles = handles, fontsize = 10, handlelength=0, handletextpad=0, markerscale=0)
        except:
            pass
   
        # Save image
        if final_map:
            fig.savefig(Path(self.skymap_folder, self.out_name+'_final_skymap.pdf'), bbox_inches = 'tight')
        else:
            fig.savefig(Path(self.skymap_folder, self.out_name+'_first_skymap.pdf'), bbox_inches = 'tight')

        # Show and close
        if show is True:
            plt.show()    
        plt.close()
    

    
    
    def evaluate_volume_map(self):
        """
        Evaluate volume map and compute credbile volumes
        """
        if not self.volume_already_evaluated:

            if self.inclination==False:
                self.vol_density = self.density
            else:
                self.vol_density = marginalise(self.density, [3])
                self.incl_density = marginalise(self.density, [0, 1, 2])

            
            #p_vol= self.vol_density._pdf_probit(self.probit_grid)*self.inv_J
            p_vol               = self.vol_density.pdf(self.grid) 
            #p_vol               = self._pdf_probit(self.probit_grid) /inv_Jacobian(self.grid)*self.inv_J
            #p_vol               = self._pdf_probit(self.probit_grid) * self.inv_J
            self.norm_p_vol     = (p_vol*np.exp(self.log_measure_3d.reshape(p_vol.shape))*self.dD*self.dra*self.ddec).sum()
            self.log_norm_p_vol = np.log(self.norm_p_vol) 
            self.p_vol          = p_vol/self.norm_p_vol
            
            
            #By default computes log(p_vol). If -infs are present, computes log_p_vol
            with np.errstate(divide='raise'):
                try:
                    self.log_p_vol = np.log(self.p_vol)
                except FloatingPointError:
                   
                    self.log_p_vol = self.density._logpdf_probit(self.probit_grid) + self.log_inv_J  - self.log_norm_p_vol
                        
            self.p_vol     = self.p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.log_p_vol = self.log_p_vol.reshape(len(self.ra), len(self.dec), len(self.dist))
            self.volume_already_evaluated = True

        self.volumes, self.idx_CR, self.volume_heights = ConfidenceVolume(self.log_p_vol, self.ra, self.dec, self.dist, log_measure = self.log_measure_3d, adLevels = self.levels)
        
        for cr, vol in zip(self.levels, self.volumes):
            self.volumes_N[cr].append(vol)
            



    def evaluate_catalog(self, final_map):
        """
        Evaluate the probability of being the host for each entry in the galaxy catalog and rank it accordingly.
        
        Arguments:
            bool final_map: flag to raise if the inference is finished
        """
        #log_p_cat = self.density._logpdf(self.cartesian_catalog) -inv_Jacobian(self.catalog)- self.log_norm_p_vol
        self.log_p_cat             = self.vol_density._logpdf_probit(self.probit_catalog) + self.log_inv_J_cat - self.log_norm_p_vol
        self.log_p_cat_to_plot     = self.log_p_cat[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        self.p_cat_to_plot         = np.exp(self.log_p_cat_to_plot)
        
        self.cat_to_plot_celestial = self.catalog[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])]
        
        self.sorted_cat            = np.c_[self.cat_to_plot_celestial[np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_cat_to_txt     = np.c_[self.catalog_with_mag[np.where(self.log_p_cat > self.volume_heights[np.where(self.levels == self.region)])][np.argsort(self.log_p_cat_to_plot)], np.sort(self.log_p_cat_to_plot)][::-1]
        self.sorted_p_cat_to_plot  = np.sort(self.p_cat_to_plot)[::-1]
        
        self.cond_cat_to_txt = []
        if self.theta_condition ==True:
            if self.max_n_gal_cond == None:
                self.sorted_cat_to_condition = self.sorted_cat_to_txt
            else:
                self.sorted_cat_to_condition = self.sorted_cat_to_txt[:self.max_n_gal_cond]
            for row in self.sorted_cat_to_condition:
                ra = row[1]
                dec = row[2]
                dL = row[3]
                ddL = row[4]

                distances = np.random.normal(dL, ddL,500)
    
                inclinations_draws = [condition(self.density,[ra, dec, distance], [0, 1, 2]) for distance in distances]
                incl_samples    = rvs_median(inclinations_draws, size = 1000)
                median          = np.median(incl_samples)
                percentile_5    = np.percentile(incl_samples, 5)
                percentile_95   = np.percentile(incl_samples, 95)
                min             = median-percentile_5
                plus            = percentile_95-median

                
                row = list(row)
               
                row.append(median)
                row.append(min)
                row.append(plus)
                self.cond_cat_to_txt.append(row)
        self.cond_cat_to_txt = np.array(self.cond_cat_to_txt)


        if final_map==True:
            np.savetxt(Path(self.hosts_folder, self.out_name+'_ranked_hosts_final.txt'), self.sorted_cat_to_txt, header = self.glade_header)#,fmt = '%.5f')
            if self.theta_condition ==True:
                np.savetxt(Path(self.hosts_folder, self.out_name+'_ranked_hosts_theta_cond_final.txt'), self.cond_cat_to_txt, header = self.glade_header_cond)#, fmt = '%.5f')
        else:
            np.savetxt(Path(self.hosts_folder, self.out_name+'_ranked_hosts_intermediate.txt'), self.sorted_cat_to_txt, header = self.glade_header)#,fmt = '%.5f')
            if self.theta_condition ==True:
                np.savetxt(Path(self.hosts_folder, self.out_name+'_ranked_hosts_theta_cond_intermediate.txt'),self.cond_cat_to_txt, header =self.glade_header_cond)#,  fmt = '%.5f')
    
    


    
    def make_volume_map(self, final_map, show = False):
            """
            Produce maps with 3D and 2D scatter plots of galaxies, if a catalog is provided.
            
            Arguments:
                bool final_map: flag to raise if the inference is finished
                int n_gals:     number of galaxies to plot
            """

            # Evaluate volume map and catalog
            n_gals = self.n_gal_to_plot
            self.evaluate_volume_map()
            if self.catalog is None:
                return
            self.evaluate_catalog(final_map)
           
            # 3D plot in celestial coordinates
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(self.cat_to_plot_celestial[:,0], self.cat_to_plot_celestial[:,1], self.cat_to_plot_celestial[:,2], c = self.p_cat_to_plot, marker = '.', alpha = 0.9, s = 3, cmap = 'Reds')
            vol_str = ['${0:.0f}\\%'.format(100*self.levels[-i])+ '\ \mathrm{CR}:'+'{0:.0f}'.format(self.volumes[-i]) + '\ \mathrm{Mpc}^3$' for i in range(len(self.volumes))]
            vol_str = '\n'.join(vol_str + ['${0}'.format(len(self.cat_to_plot_celestial)) + '\ \mathrm{galaxies}\ \mathrm{in}\ '+'{0:.0f}\\%'.format(100*self.levels[np.where(self.levels == self.region)][0])+ '\ \mathrm{CR}$'])
            ax.text2D(0.05, 0.95, vol_str, transform=ax.transAxes)
            ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$', fontsize = 11)
            ax.set_ylabel('$\\delta \ \mathrm{[rad]}$', fontsize = 11)
            ax.set_zlabel('$\mathrm{d}_{\mathrm{L}} \ \mathrm{[Mpc]}$', fontsize = 11)
            ax.tick_params(axis='both', which='major', labelsize=11)
            plt.tight_layout()
            
            # Save 
            if final_map:
                fig.savefig(Path(self.volume_folder, self.out_name+'_volume_map_final.pdf'), bbox_inches = 'tight')
            else:
                fig.savefig(Path(self.volume_folder, self.out_name+'_volume_map_intermediate.pdf'), bbox_inches = 'tight')

            # Show and close
            if show is True: 
                plt.show()    
            plt.close()
            
            
            # 2D sky projection with galaxies in the 90% credible volume 
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # plot limits
            max_level_idx = self.skymap_idx_CR[len(self.levels) - 1] 
            ra_min = np.min(self.ra[max_level_idx.T[0]])
            ra_max = np.max(self.ra[max_level_idx.T[0]])
            delta_ra = ra_max - ra_min
            dec_min = np.min(self.dec[max_level_idx.T[1]])
            dec_max = np.max(self.dec[max_level_idx.T[1]])
            delta_dec = dec_max - dec_min
            x_lim = [max(ra_min - delta_ra,0.), min(ra_max + delta_ra, 2*np.pi)] 
            y_lim = [max(dec_min - delta_dec, -np.pi/2), min(dec_max + delta_dec, np.pi/2)] 
            
            # 2D credible regions
            c1 = ax.contour(self.ra_2d, self.dec_2d, self.log_p_skymap.T, np.sort(self.skymap_heights), colors = 'black', linewidths = 0.5, linestyles = 'solid')

            # Scatterplot potential galaxy hosts
            c = ax.scatter(self.sorted_cat[:,0][:int(n_gals)], self.sorted_cat[:,1][:int(n_gals)], c = self.sorted_p_cat_to_plot[:int(n_gals)], marker = '+', cmap = 'coolwarm', linewidths = 1)
        
            # Labels and legend
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
            patch = mpatches.Patch(color='grey', label='${0}'.format(self.mix.n_pts)+'\ \mathrm{samples}$', alpha = 0)
            handles.append(patch)
            plt.colorbar(c, label = '$p_{host}$')
            ax.set_xlabel('$\\alpha \ \mathrm{[rad]}$')
            ax.set_ylabel('$\\delta \ \mathrm{[rad]}$')
    
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            try:
                ax.legend(handles = handles, loc = 2, fontsize = 10, handlelength=0, labelcolor = leg_col)
            except:
                pass
            
            # Save
            if final_map:
                fig.savefig(Path(self.skymap_folder, self.out_name+'_galaxies_final.pdf'), bbox_inches = 'tight')
            else:
                fig.savefig(Path(self.skymap_folder, self.out_name+'_galaxies_intermediate.pdf'), bbox_inches = 'tight')  

            # Show and close 
            if show is True: 
                plt.show()    
            plt.close()

        






    def make_entropy_plot(self, show = False):
        """
        If entropy == True, produces entropy and angular coefficient plots.
        """
        fig, axs = plt.subplots(2, 1, sharex=True)

        axs[0].plot(np.arange(len(self.R_S))*self.entropy_step, (self.R_S), color = 'steelblue', lw = 0.7)
        axs[0].set_ylabel('$S(N)\ [\mathrm{bits}]$')
        axs[0].grid(True)

        axs[1].axhline(0, lw = 0.5, ls = '--', color = 'r')
        axs[1].plot(np.arange(len(self.ac))*self.entropy_step + self.entropy_ac_steps, self.ac, color = 'steelblue', lw = 0.7)
        axs[1].set_xlabel('$N$')
        axs[1].set_ylabel('dS/dN')
        axs[1].grid(True)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        axs[1].yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        
        fig.savefig(Path(self.entropy_folder, self.out_name + '_entropy.pdf'), bbox_inches = 'tight')

        if show is True: 
            plt.show()
        plt.close()





    

    def plot_samples(self, samples, final_map = False, show = False):
        """
        Draw samples from the inferred distribution and plot them.
        
        Arguments:
            array samples: a (num,3) array containing num samples of (dl, ra, dec) or a (num,4) array containing num samples of (dl, ra, dec, theta_jn)
        """

        # samples to plot
        samples = np.array(samples)   
        samples_from_DPGMM = self.density.rvs(len(samples)) 

        # truth values
        if self.true_host is not None:
            if self.inclination:
                if self.true_inclination is None:
                    truth = [self.true_host[0], self.true_host[1], self.true_host[2], None] 
                else:
                    truth = [self.true_host[0], self.true_host[1], self.true_host[2], self.true_inclination[0]] 
            else:
                truth = self.true_host
        else:
            truth = None

        #corner plot limits
        if self.inclination is True:
            dim = 4
        else:
            dim = 3  
            
        limc=[]
        for i in range(dim):
            perc_1 =   np.percentile(samples_from_DPGMM[:,i], 1)
            perc_99 =  np.percentile(samples_from_DPGMM[:,i], 99)
            delta_perc = perc_99 - perc_1
            limc.append([perc_1  - 0.2*delta_perc, perc_99 + 0.2*delta_perc ])       

        #plot
        c = corner(samples, range=limc, color = 'black', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{Samples}$'}, truths = truth, truth_color = 'black', quiet = True)
        c = corner(samples_from_DPGMM, fig = c, range=limc, color = 'dodgerblue', labels = self.labels, hist_kwargs={'density':True, 'label':'$\mathrm{DPGMM}$'},  quiet = True)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.15, 0., 0.5, 1.4),frameon = False,fontsize = 15)
        
        #save
        if final_map==True:
            c.savefig(Path(self.corner_folder, self.out_name+'_final.png'))
        else:
            c.savefig(Path(self.corner_folder, self.out_name+'_intermediate.png'))

        #show and close
        if show is True:
            plt.show()    
        plt.close()   

    


    def save_density(self, final_map = False):
        """
        Build and save density
        """
        density = self.mix.build_mixture()
        if final_map == False:
            save_density([density], folder = self.density_folder, name  = self.out_name +f'_intermediate', ext = 'json')
        else:
            save_density([density], folder = self.density_folder, name  = self.out_name +f'_final', ext = 'json')




    
    def save_log(self):
        """
        Save log with important information
        """
        with open(Path(self.log_folder, self.out_name +f'_log.json'), 'w') as dill_file:
            json.dump(self.log_dict, dill_file)

    
    def save_CR(self):
        """
        Save txt file with sky and volume credible regions
        """
       
        with open(Path(self.CR_folder, self.out_name +f'_areas.json'), 'w') as dill_file:
            json.dump(self.areas_N, dill_file)

        with open(Path(self.CR_folder, self.out_name +f'_volumes.json'), 'w') as dill_file:
            json.dump(self.volumes_N, dill_file)







    def inclination_histogram(self, final_map):
        incl_samples = self.incl_density.rvs(5000)
        median          = np.median(incl_samples)
        percentile_5    = np.percentile(incl_samples, 5)
        percentile_95   = np.percentile(incl_samples, 95)
        min             = median-percentile_5
        plus            =  percentile_95-median
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(incl_samples,color = 'dodgerblue', density = True, histtype = 'step')
        ax.axvline(percentile_5)
        ax.axvline(median, color = 'red',label = 'median' )
        ax.axvline(percentile_95)
        ax.set_title('$\\theta_{jn}$ ='+ f'{median:.2f}' + f'+${plus:.2f}-{min:.2f}$')
        if self.true_inclination is not None:
            ax.axvline(self.true_inclination, label = 'True', color = 'black')
        ax.legend()
        
        #header_to_print = "median  +  - \n"
        #inclination_to_print = np.array([median, plus, min])
        #if final_map==True:
        #    fig.savefig(Path(self.inclination_folder, self.out_name + '_theta_jn_final.pdf'), bbox_inches = 'tight')
        #    np.savetxt(Path(self.inclination_folder,self.out_name + '_theta_jn_final.txt' ),inclination_to_print,  header = header_to_print , newline = '')
        #else:
        #    fig.savefig(Path(self.inclination_folder, self.out_name + '_theta_jn_intermediate.pdf'), bbox_inches = 'tight')
        #    np.savetxt(Path(self.inclination_folder,self.out_name + '_theta_jn_intermediate.txt' ), inclination_to_print, header = header_to_print, newline = '')
        plt.close()    




        


    def intermediate_skymap(self, sample, sampling_time = None, show = False):
        """
        Add a sample to the mixture, computes the entropy (if entropy == True), and releases an intermediate skymap as soon as convergence is reached.

        Arguments:
            3D or 4D array sample
        """
        if sampling_time is not None:
            self.log_dict['sampling_time'] = sampling_time
        self.sampling_time = sampling_time
        self.mix.add_new_point(sample)
        self.log_dict['total_samples'] = self.mix.n_pts
        self.samples.append(sample)
        self.i +=1
        self.N_PT.append(self.mix.n_pts)
        self.N_clu.append(self.mix.n_cl)
        if self.entropy:
            if self.i%self.entropy_step == 0:
                R_S = compute_entropy_single_draw(self.mix, self.n_entropy_MC_draws)
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
                            print('INTERMEDIATE RECONSTRUCTION')
                            if sampling_time is not None:
                                self.log_dict['first_skymap_time'] = sampling_time
                            self.density = self.mix.build_mixture()    
                            self.log_dict['first_skymap_samples'] = self.mix.n_pts
                            self.plot_samples(self.samples, final_map = False, show = show)
                            self.make_skymap(final_map = False, show = show)
                            self.make_volume_map(final_map = False, show = show)
                            self.save_density(final_map = False)
                            self.flag_skymap = True
                            
                            #if self.inclination==True: 
                            #    self.inclination_histogram(final_map = False)
                            
                    self.ac.append(ac)
       # self.save_log()
        
        



    def initialise(self): 
        """
        Initialise the existing instance of the skyfast class to new initial conditions. 
        This could be useful to analyze multiple times a GW event without the need of initializing skyfast from scratch every time.
        """    

        del self.mix
        self.mix = DPGMM(self.bounds, prior_pars= self.prior_pars, alpha0 = 1, probit = True)
        self.samples = []
        self.R_S = []
        self.ac = []
        self.volume_already_evaluated = False
        self.ac_cntr = self.n_sign_changes
        self.i = 0
        self.flag_skymap = False






               

