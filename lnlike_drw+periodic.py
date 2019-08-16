def lnlike_periodic(theta, fit_time, fit_signal, fit_error):
    
    t_ratio, t_shift, s_ratio, s_shift, drw_sigma, drw_tau = theta
    
    fit = np.interp( fit_time, (sim_time*t_ratio)+t_shift, (sim_signal*s_ratio)+s_shift )
    model = fit_signal - fit
    
    cov_D     = np.zeros([len(fit_signal),  len(fit_signal)])
    cov_sigma = np.zeros([len(fit_signal),  len(fit_signal)]) 
    
    
    for i in range(len(fit_time)):
        for j in range(len(fit_time)):
            delta_t = np.abs(fit_time[i] - fit_time[j])
            cov_D[i,j] = drw_sigma**2.0 * np.exp(- delta_t / drw_tau)
            
            if (i==j):
                cov_sigma[i,j] = fit_error[i]**2.0
    
    #
    cov         = cov_D + cov_sigma
    cov_inverse = np.linalg.inv(cov)
    chi_square  = np.dot( model, np.dot(cov_inverse, model) )
        
    (sign, logdet) = np.linalg.slogdet( cov_D+cov_sigma )
    
    lnlikeli = -0.5*logdet - 0.5*chi_square
    
        
    return lnlikeli
    
    

def lnlike_drw(theta, fit_time, fit_signal, fit_error):
    
    sigma, tau, mean_mag = theta
    
    model = fit_signal - mean_mag
    
    cov_D     = np.zeros([len(signal),  len(signal)])
    cov_sigma = np.zeros([len(signal),  len(signal)]) 
    
    
    for i in range(len(fit_time)):
        for j in range(len(fit_time)):
            delta_t = np.abs(fit_time[i] - fit_time[j])
            cov_D[i,j] = sigma**2.0 * np.exp(- delta_t / tau)
            
            if (i==j):
                cov_sigma[i,j] = fit_error[i]**2.0
    
    #
    cov         = cov_D + cov_sigma
    cov_inverse = np.linalg.inv(cov)
    chi_square  = np.dot( model, np.dot(cov_inverse, model) )
        
    (sign, logdet) = np.linalg.slogdet( cov_D+cov_sigma )
    
    lnlikeli = -0.5*logdet - 0.5*chi_square
    
        
    return lnlikeli