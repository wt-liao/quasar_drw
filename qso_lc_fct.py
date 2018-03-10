import numpy as np
from numpy import sqrt
from astropy.table import Table
#from PyAstronomy import pyasl
from qso_fit_fix import qso_engine
from scipy.optimize import curve_fit
from astropy.io import fits

## ======== damped random walk model v2 ======== ##
def drw_lc2(t, tau, c):
    """
    formula from Kelly+2009 eqn(1):
    mean value: mean * tau
    variance: tau*sigma^2/2
    
    input:
    t: time sequence (increasing)
    tau: time scale for drw 
    sigma: related to variance 
    mean: related to mean value of lc (it's 'b' in Kelly+2009)
    
    output:
    X: light curve 
    
    """
    max_t = np.max(t)
    min_t = np.min(t)
    
    N = int((max_t-min_t)+1.)
    time = np.linspace(min_t, max_t, N)
    
    
    X = np.zeros([N])
    X_return = np.zeros([len(t)])
    
    X[0] = 0.
    #X[0] = np.random.standard_normal()
    np.random.seed()
    rand = np.random.standard_normal(N)

    for i in range(1, N):
        #dt = time[i] - time[i-1]
        dt = 1.
        X[i] = X[i-1] - (1./tau)*X[i-1]*dt + sqrt(c)*sqrt(dt)*rand[i] 
    
    for j in range(1, len(t)):
        ind = np.argmin( np.abs(t[j]-time) )
        X_return[j] = X[ind]
    #X -= np.mean(X)
        
    return X_return
## ======== END ======== ##

## ======== G2015 light curve ======== ##
# read in G2015 light curve from an illed-structure csv list
def get_G2015_lc():
    filepath = '../data/G2015_list.csv'
    data = Table.read(filepath, format='ascii')

    # read in data from csv 
    input_name = []
    CTRS_ID    = []
    mag_all = []
    mjd_all = []
    err_all = []
    ra_all  = []
    dec_all = []
    z_all   = []

    count = 0
    name_buffer = data.field('InputID')[0]
    mag = []
    mjd = []
    err = []
    ra_all.append(data.field('RA')[0])
    dec_all.append(data.field('Decl')[0])
    CTRS_ID.append(data.field('ID')[0])
    input_name.append(data.field('InputID')[0])

    while ( count < len(data.field('RA') ) ):
        
    
        if  data.field('InputID')[count] != name_buffer:
        
            mag_all.append(mag)
            mjd_all.append(mjd)
            err_all.append(err)
        
            name_buffer = data.field('InputID')[count]
        
            mag = []
            mjd = []
            err = []
            ra_all.append(data.field('RA')[count])
            dec_all.append(data.field('Decl')[count])
            CTRS_ID.append(data.field('ID')[count])
            input_name.append(data.field('InputID')[count])
    
        if data.field('InputID')[count] == name_buffer:
            mag.append( format(data.field('Mag')[count], '.8') )
            err.append( format(data.field('Magerr')[count], '.8') )
            mjd.append( format(data.field('MJD')[count], '.10') )
        
        count += 1
    
    # for the last light curve
    mag_all.append(mag)
    mjd_all.append(mjd)
    err_all.append(err)
        
    return CTRS_ID, ra_all, dec_all, mjd_all, mag_all, err_all
## ======== END G2015 light curve ======== ##


## ======== Zheng2015 light curve ======== ##
def get_Zheng2015_lc():
    filepath = '../data/Z2015_CRTS.csv'
    data = Table.read(filepath, format='ascii')

    time   = data.field('MJD')
    signal = data.field('Mag')
    error  = data.field('Magerr')
    RA     = data.field('RA')[0]
    Dec    = data.field('Dec')[0]
    
    return time, signal, error, RA, Dec
## ======== END Zheng2015 light curve ======== ##


## ======== PanSTARRS light curve ======== ##
def get_PS1_lc(field, band, lc_num):
    ## field:  which field - 03 - 10
    ## band:   which band - g, r, i, z
    ## lc_num: which light number in this field
    
    filepath  = '../data/light_curve_PanSTARRS/PS1MD' + format(field, '02d') + '_20140912.fits'
    data = fits.open(filepath)[1].data
    
    time   = data.field( 'lc_mjd_' + str(band) )[lc_num]
    signal = data.field( 'lc_mag_' + str(band) )[lc_num]
    error  = data.field( 'lc_err_' + str(band) )[lc_num]
    
    RA     = data.field('ra')[lc_num]
    Dec    = data.field('dec')[lc_num]
    
    return time, signal, error, RA, Dec
## ======== END Zheng2015 light curve ======== ##


## ======== Stripe82 light curve ======== ##
def get_Liu16_S82(filename, band):
    # band: "u", "g", "r", "i", "z"
    
    if (band != "g") and (band != "r") and (band != "i"):
        print ("choose the right band: g, r, i")
    
    filepath = "../data/Liu16_S82/" + str(filename)
    
    # get title
    f    = open(filepath, "r")
    f.readline()
    f.readline()
    col_name = f.readline().split(",")
    col_name = np.array(col_name)
    f.close()
    
    data = np.loadtxt(filepath, delimiter=",")
    
    # create an array for index searching
    ind_array = np.linspace(0, len(col_name)-1, len(col_name))
    
    ind_ra  = (col_name == 'ra')
    ind_ra  = int( ind_array[ind_ra] )
    
    ind_dec = (col_name == 'dec')
    ind_dec = int( ind_array[ind_dec] )
    
    ind_signal = (col_name == str(band))
    ind_signal = int( ind_array[ind_signal] )
    
    ind_error = ( col_name == "err_" + str(band) )
    ind_error = int( ind_array[ind_error] )
    
    ind_time = ( col_name == "mjd_" + str(band) )
    ind_time = int( ind_array[ind_time] )

    ra     = data[0, ind_ra]
    dec    = data[0, ind_dec]
    signal = data[:, ind_signal]
    error  = data[:, ind_error]
    time   = data[:, ind_time]
    
    return time, signal, error, ra, dec

## ======== END stripe82 ======== ##


########################################
## ======== pre-process data ======== ##
########################################

def mag2flux(signal, error):
    flux = 10.**(-1.*signal/2.5)
    return 10.**(-1.*signal/2.5), np.abs( -flux*error*np.log(10.)/2.5 )

def flux2mag(signal, error):
    return -2.5*np.log10(signal), np.abs( -2.5* error/signal/np.log(10.))

def weighted_mean(signal, error):
    signal_mean = np.sum(signal/error**2.) / np.sum(1./error**2.) 
    #error_mean  = np.sqrt( 1. / np.sum(1./error**2.)  ) 
    error_mean  = np.sqrt( np.sum(error**2.) ) / np.sqrt( np.float(len(signal)) )
    return signal_mean, error_mean
    
def pre_process(time, signal, error):
    # this only sort data in increasing time sequence
    
    inds = (error > 0.)
    time   = time[inds]
    signal = signal[inds]
    error  = error[inds]
    
    inds = np.argsort(time)
    time   = time[inds]
    signal = signal[inds]
    error  = error[inds]

    return time, signal, error

def no_outlier(time, signal, error):
    # use 3 point median filter 
    #signal = medfilt(signal)
    
    n_point_old = 0
    n_point_new = len(signal)
    
    i = 0
    while n_point_new != n_point_old and i<=50 and n_point_new > 7.:
        n_point_old = len(signal)
    
        inds = pyasl.polyResOutlier(time, signal, deg=4, stdlim=2.5)
    
        time   = time[inds[0]]
        signal = signal[inds[0]]
        error  = error[inds[0]]
        n_point_new = len(signal)
        
        i += 1
    
    return time, signal, error

def bin_data(time, signal, error):
    time2   = []
    signal2 = []
    error2  = []
    
    count = 0
    
    while(count < len(time)):
        inds = ( np.floor(time) == np.floor(time[count]) )
        signal_temp = signal[inds]
        error_temp  = error[inds]
        nn = len(signal_temp)
        
        signal_temp, error_temp = mag2flux(signal_temp, error_temp)
        signal_temp, error_temp = weighted_mean(signal_temp, error_temp)
        signal_temp, error_temp = flux2mag(signal_temp, error_temp)
        
        time2.append( np.floor(time[count]) ) 
        signal2.append(signal_temp)
        error2.append(error_temp)
        
        count += nn
        
    time2   = np.asarray(time2)
    signal2 = np.asarray(signal2)
    error2  = np.asarray(error2)
    
    return time2, signal2, error2

##############################################
## ======== END preprocess package ======== ##
##############################################

## ======== fit sine curve ======== ##
def sin_curve(time, signal):
    
    def fun_fit(t, w, phi, A, A0):
        return A*np.sin(w*t + phi) + A0
        
    popt, pcov = curve_fit(fun_fit, time, signal)
    
    # reconstruct sin curve
    w   = popt[0]
    phi = popt[1]
    A   = popt[2]
    A0  = popt[3]

    fit_sin = np.zeros([100])
    fit_t = np.linspace(np.min(time), np.max(time), 100)
    fit_sin = A*np.sin(w*fit_t +phi) + A0
    
    #return fit_t, fit_sin
    return A, A0, w, phi, fit_t, fit_sin
## ======== END sine curve fitting ======== ##

def write4zdcf(time, signal, error, lc="lc"):
    f = open(lc, "w")
    for i in range(len(time)):
        f.write(format(time[i], '.8e') + '\t' + format(signal[i], '.8e') + '\t' + format(error[i], '.8e') + '\n')
    f.close()
    
def zdcf_result(filename):
    f_zdcf = np.loadtxt(filename)
    zdcf_tau = f_zdcf[:, 0]

    zdcf_tau_err = []
    zdcf_tau_err.append(f_zdcf[:, 1])
    zdcf_tau_err.append(f_zdcf[:, 2])

    zdcf = f_zdcf[:, 3]
    zdcf_err = []
    zdcf_err.append(f_zdcf[:, 4])
    zdcf_err.append(f_zdcf[:, 5])
    
    n = f_zdcf[:, 6]
    
    return zdcf_tau, zdcf_tau_err, zdcf, zdcf_err, n

    
