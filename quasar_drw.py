import numpy as np
from numpy import mean, median
from astropy.stats import sigma_clip
import scipy.signal as sig
#from lomb_scargle_red_fix import lomb
#from astroML.time_series import lomb_scargle


class quasar_drw:
    
    def __init__(self, time, signal, error, redshift, preprocess=True):
        self.time     = np.array(time, dtype=np.float32)
        self.signal   = np.array(signal, dtype=np.float32)
        self.error    = np.array(error, dtype=np.float32)
        self.redshift = float(redshift)
        self._preprocessed = False
        
        if ( len(time) != len(signal) ) or ( len(time)!= len(error) ):
            print("[quasar_drw] Error in input data: time, signal, error must have the same length.")
        
        if preprocess == True:
            self._preprocess()
        
        ## parameters for periodogram
        self.__Tspan    = float( np.max(self.signal) - np.min(self.signal) )
        self.__Ndata     = len(self.signal)
        self.__psd_freq = \
            np.linspace(1.0/self.__Tspan, self.__Ndata/(2.0*self.__Tspan), self.__Ndata) 
        self.__dt = self.__Tspan / float(self.__Ndata)
        self.__df = self.__psd_freq[1] - self.__psd_freq[0]
        
        
    def _preprocess(self):
        self._sort_data()
        self._no_outlier()
        self._bin_data()
        self._preprocessed = True
    
    
    def get_lc(self):
        """ output: time, signal, error """
        return (self.time, self.signal, self.error)
        
    def get_redshift(self):
        return self.redshift
        
    
    def ls_periodogram(self):
        """ calculate periodogram using Lomb-Scargle periodogram from scipy """
        time   = self.time
        signal = self.signal
        # power spectrum
        ps  = sig.lombscargle(time, signal-np.mean(signal), self.__psd_freq*(2.0*np.pi) )
        # power spectrum density
        psd = 2.0*ps/self.__df
        
        return psd
    
    
    def periodogram_err(self, N_bootstrap = 1000):
        """ estimate error of scipy-periodogram using bootstrap """
        bootstrap_all = np.zeros([len(self.__psd_freq), N_bootstrap])
        
        for i in range(N_bootstrap):
            idx = np.random.random_integers(low =0, high=len(self.time)-1, size=len(self.time))
            idx = np.sort(idx)
        
            time_bootstrap   = self.time[idx]
            signal_bootstrap = self.signal[idx]
        
            psd = self.ls_periodogram(time_bootstrap, signal_bootstrap)
            bootstrap_all[:, i] = psd
        
        # record 1-sigma level
        bootstrap_result = np.percentile(bootstrap_all, [16, 50, 84], axis=1)
        yerr = np.abs(bootstrap_result[0, :] - bootstrap_result[2, :])
        
        # deal with 0 error 
        idx = (yerr == 0.0)
        yerr[idx] = 1.0e-7
    
        return yerr
        
    
    def ls_dered(self, log10_tau, log10_var, do_fit=False):
        """ 
        calculate periodogram using de-redening Lomb-Scargle periodogram from Zheng et al. (2016) 
        Their code can be download from here: http://butler.lab.asu.edu/qso_period/
        N.B. This package requires Python 2. 
        """
        psd, lvar, ltau = \
            lomb(self.time, self.signal, self.error, self.__psd_freq[1], self.__df, self.__Ndata, \
                 ltau=log10_tau, lvar=log10_var, do_fit = do_fit)
        return psd
         
        
    def ls_astroML(self):
        """
        calculate periodogram using generalized Lomb-Scargle periodogram from AstroML
        function description: http://www.astroml.org/modules/generated/astroML.time_series.lomb_scargle.html
        example: http://www.astroml.org/book_figures/chapter10/fig_LS_example.html
        """
        
        pass
        
    
    def zdcf(self):
        pass
    
    
    def fit_drw_emcee(self, nwalker=500, burnin=100):
        ndim    = 2
        nwalker = nwalker
        pos     = []
        
        psd      = self.ls_periodogram()
        psd_err  = self.periodogram_err()
        psd_freq = self.__psd_freq
        z        = self.redshift
        
        # initiate a gaussian distribution aroun dthe mean value
        tau_sample = np.random.lognormal(mean=np.log(300.), sigma=1.15, size=nwalkers)
        c_sample   = np.random.lognormal(mean=np.log(0.008**2.), sigma=0.5, size=nwalkers)
        
        tau_sample, c_sample = np.log(tau_sample), np.log(c_sample)
        
        for i in range(nwalkers):
            parameter = np.array([tau_sample[i], c_sample[i]])
            pos.append(parameter)
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(psd_freq, psd, psd_err, z), a=4.0)
        
        # start MCMC
        sampler.run_mcmc(pos, 500)
    
        # remove burn-in
        burnin = burnin
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

        #tau_range   = np.percentile(samples, [5, 50, 95], axis=0)[:,0]
        #c_range = np.percentile(samples, [5, 50, 95], axis=0)[:,1]
        return samples
        
        
    def flux2mag(self):
        """ convert flux to mag """
        flux = self.signal
        self.signal = 22.5 - 2.5 * np.log10(flux)
        self.error  = np.abs( -2.5* self.error/flux/np.log(10.))
    
    
    ### ********************************* ###
    ###  helper functions for preprocess  ###
    ### ********************************* ###
        
    def _sort_data(self):
        
        # take away points w/o data
        idx = (self.error > 0.)
        time   = self.time[idx]
        signal = self.signal[idx]
        error  = self.error[idx]
        
        # sort
        idx = np.argsort(time)
        time   = time[idx]
        signal = signal[idx]
        error  = error[idx]
        
        # restore data
        self.time   = time
        self.signal = signal
        self.error  = error
    
    
    def _no_outlier(self, iters=100):
        after_clip = sigma_clip(self.signal, sigma=2, iters=iters, cenfunc=mean, copy=True)
        
        idx = ~(after_clip.mask)
        self.time   = self.time[idx]
        self.signal = self.signal[idx]
        self.error  = self.error[idx]

        
    def _bin_data(self):
        time2   = []
        signal2 = []
        error2  = []
        count   = 0
    
        while(count < len(self.time)):
            idx = ( np.floor(self.time) == np.floor(self.time[count]) )
            signal_temp = self.signal[idx]
            error_temp  = self.error[idx]
            nn          = len(signal_temp)
        
            signal_temp, error_temp = self.__mag2flux(signal_temp, error_temp)
            signal_temp, error_temp = self.__weighted_mean(signal_temp, error_temp)
            signal_temp, error_temp = self.__flux2mag(signal_temp, error_temp)
        
            time2.append( np.floor(self.time[count]) ) 
            signal2.append( signal_temp )
            error2.append( error_temp )
        
            count += nn
        
        self.time   = np.asarray(time2)
        self.signal = np.asarray(signal2)
        self.error  = np.asarray(error2)
    

    def __mag2flux(self, signal, error):
        flux = 10.**(-1.*signal/2.5)
        return 10.**(-1.*signal/2.5), np.abs( -flux*error*np.log(10.)/2.5 )
    
    
    def __flux2mag(self, signal, error):
        return -2.5*np.log10(signal), np.abs( -2.5* error/signal/np.log(10.))
        
    
    def __weighted_mean(self, signal, error):
        signal_mean = np.sum(signal/error**2.) / np.sum(1./error**2.) 
        error_mean  = np.sqrt( np.sum(error**2.) ) / np.sqrt( np.float(len(signal)) )
        return signal_mean, error_mean
    
    ### *********************************** ###
    ###  END of helper func for preprocess  ###
    ### *********************************** ###
    
    
    
    
    ### ************************* ###
    ### helper functions for zdcf ###
    ### ************************* ###
    def __write4zdcf(self):
        pass
    
        
    def __read_zdcf_result(self):
        pass
    
    
    ### *************************** ###
    ### END of helper func for zdcf ###
    ### *************************** ###
    
    
    ##### ------------------------------- #####
    ##### --- END of quasar_drw class --- #####
    ##### ------------------------------- #####




##### ------------------------------------ #####
##### --- Class: light curve generator --- #####
##### ------------------------------------ #####

class lc_generator:
    """ generate mock light curve based on damped random walk model """
    
    def __init__(self, lc_drw, tau, c):
        self.lc  = lc_drw   # lc_drw is a quasar_drw object
        self.tau = float(tau)
        self.c   = float(c)
        
        if (self.lc._preprocessed == False):
            self.lc._preprocess()
        
        
    def lc_gen(self):
        time, signal, error = self.lc.get_lc()
        redshift            = self.lc.get_redshift()
        
        N = int(np.max(time) - np.min(time) + 1)
        t_array = np.linspace(np.min(time), np.max(time), N)
        
        mock_lc = np.zeros(N, dtype=np.float32)
        
        idx_array = np.zeros(N, dtype=np.bool)
        idx_array[0] = True
        
        np.random.seed()
        rand = np.random.standard_normal(N)
        
        dt = 1.0 # for preprocessed data
        for i in range(1, N):
            mock_lc[i] = \
                mock_lc[i-1] - (1./self.tau)*mock_lc[i-1]*dt + np.sqrt(self.c)*np.sqrt(dt)*rand[i]
            
            idx = (time == t_array[i])
            
            if idx.any():
                idx_array[i] = True
        
        #lc_out = quasar_drw(time, signal, error, redshift, preprocess=False)
        lc_out = quasar_drw(time, mock_lc[idx_array]+np.mean(signal), error, redshift, preprocess=False)
        
        return lc_out
        
    