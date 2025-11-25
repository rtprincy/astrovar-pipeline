import scipy.stats as stats
import numpy as np
from astropy.timeseries import LombScargle
from astropy.timeseries.periodograms.lombscargle import utils

class LCStatistics:
    def __init__(self, mag, magerr, Time=None, lspower=None,psi=None, frequencies=None, flux=None, flux_err=None, best_freq=None,band='G'):
        """
        Initializes the class with input data.
        
        Parameters:
        mag (array): Magnitude.
        magerr (array): Magnitude error.
        Time (array): Time.
        psi (array): Hybrid psi-statistic periodogram (2*lsp/theta).
        frequencies (array): Frequencies used to compute the periodogram .
        flux (array): Flux.
        flux_err (array): Flux error.
        best_freq (float): Best frequency value (dominant frequency).
        """
        self.mag = mag
        self.magerr = magerr
        self.Time = Time
        self.lspower=lspower
        self.psi = psi
        self.psi_norm=psi/psi.max()
        self.frequencies = frequencies
        self.flux = flux
        self.flux_err = flux_err
        self.best_freq = best_freq
        self.band = band

    def ptp_amplitude(self,n_bins = 10,return_binned_mag=False):
        period=1/self.best_freq
        phase=(self.Time/period)%1    
        idx_sort=np.argsort(phase)
    
        phase=phase[idx_sort]
        mag_sorted=self.mag[idx_sort]
        magerr_sorted=self.magerr[idx_sort]
     
        
        bins = np.linspace(0, 1, n_bins+1)
    
        binned_phase = (bins[1:] + bins[:-1])/2
        
        binned_mag=np.zeros(n_bins)
        binned_magerr=np.zeros(n_bins)
        
        for j in range(n_bins):
            
            mag_bin=mag_sorted[(phase >= bins[j]) & (phase < bins[j+1])]
            magerr_bin=magerr_sorted[(phase >= bins[j]) & (phase < bins[j+1])]
    
            if mag_bin.size>0:
                
                    binned_mag[j]=np.median(mag_bin)
                    binned_magerr[j]=np.median(magerr_bin)
                
            else:
                    binned_mag[j]=np.nan
                    binned_magerr[j]=np.nan
                    
        amplitude = np.nanmax(binned_mag) - np.nanmin(binned_mag)
        
    
        if return_binned_mag:

        
            return amplitude,binned_mag, binned_magerr
            
        else:
            
            return amplitude
        

    def stetson_j(self):
        n = len(self.mag)
        if n < 2:
            return 0
        mean_mag = np.mean(self.mag)
        delta = np.sqrt(n / (n - 1)) * (self.mag - mean_mag) / self.magerr
        J = np.sum(np.sign(delta[:-1] * delta[1:]) * np.abs(delta[:-1] * delta[1:])) / n
        return J

    def std_over_rms(self):
        mag_mean = np.mean(self.mag)
        snr = np.sqrt(sum((self.mag - mag_mean) ** 2) / sum(self.magerr ** 2))
        return snr

    def max_abs_dev_from_median(self):
        median = np.median(self.mag)
        abs_deviations = np.abs(self.mag - median)
        normalized_deviations = abs_deviations / self.magerr
        return np.max(normalized_deviations)

    def abbe_value(self):
        if self.Time is None:
            raise ValueError("Time data is required for abbe_value computation.")
        ysorted = self.mag[np.argsort(self.Time)]
        abbe = sum((ysorted[1:] - ysorted[:-1]) ** 2) / (2 * sum((ysorted - np.mean(ysorted)) ** 2))
        return abbe

    def amplitude(self):
        if self.Time is None or self.best_freq is None:
            raise ValueError("Time and best_freq are required for amplitude computation.")
        lsp = LombScargle(t=self.Time, y=self.mag, dy=self.magerr, nterms=1)
        thetas = lsp.model_parameters(frequency=self.best_freq)
        phi = np.arctan(thetas[2] / thetas[1])
        amplitude = thetas[1] / np.cos(phi)
        return amplitude
        
    def rms(self):
        if self.Time is None or self.best_freq is None:
            raise ValueError("Time and best_freq are required for amplitude computation.")
        lsp = LombScargle(t=self.Time, y=self.mag, dy=self.magerr, nterms=1)
        period=1/self.best_freq
        phase=(self.Time/period)%1 
        mag_model=lsp.model(phase / self.best_freq, self.best_freq)
        res=mag_model-self.mag
        weighted_residuals=res/self.magerr      
        rms=np.sqrt(np.mean(weighted_residuals ** 2))
    
        return rms

    def rms_over_ptp_amp(self):
        ratio=self.rms()/self.ptp_amplitude()
        return ratio

    def magnitude_range(self):
        return self.mag.ptp()

    def iqr(self):
        q3, q1 = np.percentile(self.mag, [75, 25])
        return q3 - q1

    def log_sigvar(self):
        if self.flux is None or self.flux_err is None:
            raise ValueError("Flux and flux_err are required for log_sigvar computation.")
        w = 1 / (self.flux_err ** 2)
        n = self.flux.size
        w_avg = np.mean(w)
        if n > 1:
            neta = (n / (n - 1)) * (np.mean(w * (self.flux ** 2)) - (np.mean(w * self.flux) ** 2) / w_avg)
        else:
            neta = np.nan
        return np.log10(neta)

    def Kurtosis(self):
        if self.psi is None:
            raise ValueError("x is required for kurtosis computation.")
        if len(self.psi)>1:
            y=stats.kurtosis(self.psi)
        else:
            y=np.nan
        return y


    def p90_95_99(self):
        if self.psi is None:
            raise ValueError("psi is required for p95_100 computation.")
        psi_copy = np.copy(self.psi_norm)
        idx_best = np.argmax(psi_copy)
        psi_copy[max(0, idx_best - 10):min(idx_best + 10, len(self.psi))] = 0
        idx_ = np.zeros(100, dtype=int)
        for j in range(100):
            idx_best = np.argmax(psi_copy)
            idx_[j] = idx_best    
            psi_copy[max(0, idx_best - 10):min(idx_best + 10, len(self.psi_norm))] = 0
        _, p90_100 = np.percentile(self.psi_norm[idx_], [10, 90])
        _, p95_100 = np.percentile(self.psi_norm[idx_], [5, 95])
        _, p99_100 = np.percentile(self.psi_norm[idx_], [1, 99])
        return [p90_100,p95_100,p99_100]

    def p99(self):
        if self.psi is None:
            raise ValueError("psi is required for p99 computation.")
        
        _, p99 = np.percentile(self.psi_norm, [1, 99])
        return p99

    def n05(self):
        if self.psi is None:
            raise ValueError("psi is required for n05 computation.")
        
        return len(self.psi_norm[self.psi_norm >= 0.5])

    def mad(self):
        return stats.median_abs_deviation(self.mag, scale="normal")

    def psi_sigvar(self):
        if self.frequencies is None or self.psi is None:
            raise ValueError("frequencies and psi are required for psi_sigvar computation.")
         
        mad = stats.median_abs_deviation(np.array([[self.frequencies, self.psi_norm]]), axis=None)

        w = 1 / (mad ** 2)
     
        n = self.psi_norm.size
     
        w_avg = np.mean(w)
     
        if n > 1:
            neta = (n / (n - 1)) * (np.mean(w * (self.psi_norm ** 2)) - (np.mean(w * self.psi_norm) ** 2) / w_avg)
        else:
            neta = np.nan
     
        return neta

    def fap(self):
        lsp = LombScargle(t=self.Time, y=self.mag, dy=self.magerr, nterms=1)
        chi2 = utils.compute_chi2_ref(y=self.mag, dy=self.magerr)
        lsp_standard = utils.convert_normalization(Z=self.lspower, N=len(self.mag),
                                                   from_normalization='psd',
                                                   to_normalization='standard',
                                                   chi2_ref=chi2)
        fap = lsp.false_alarm_probability(lsp_standard.max(), samples_per_peak=10,
                                          minimum_frequency=self.frequencies.min(), maximum_frequency=self.frequencies.max())
     
        return fap
    

    def compute_all_parameters(self):
        """
        Compute all available parameters and store them in a dictionary.
        
        Returns:
        dict: A dictionary containing all computed parameter values.
        """
        results = {}
    
        # Compute each parameter, catching exceptions for optional data
        # try:
        #     results['stetson_j'] = self.stetson_j()
        # except Exception as e:
        #     results['stetson_j'] = np.nan
    
        # try:
        #     results['std_over_rms'] = self.std_over_rms()
        # except Exception as e:
        #     results['std_over_rms'] = np.nan
    
        # try:
        #     results['max_abs_dev_from_median'] = self.max_abs_dev_from_median()
        # except Exception as e:
        #     results['max_abs_dev_from_median'] = np.nan
    
        # try:
        #     results['abbe_value'] = self.abbe_value()
        # except Exception as e:
        #     results['abbe_value'] = np.nan
    
        try:
            results[f'amp_{self.band}'] = np.abs(self.amplitude())
        except Exception as e:
            results[f'amp_{self.band}'] = np.nan
        try:
            results[f'fap_{self.band}'] = self.fap()
        except Exception as e:
            results[f'fap_{self.band}'] = np.nan
    
        # try:
        #     results['magnitude_range'] = self.magnitude_range()
        # except Exception as e:
        #     results['magnitude_range'] = np.nan
    
        # try:
        #     results['iqr'] = self.iqr()
        # except Exception as e:
        #     results['iqr'] = np.nan

        if self.band=='G':
            try:
                results['log_sigvar'] = self.log_sigvar()
            except Exception as e:
                results['log_sigvar'] = np.nan
        
            try:
                results['p90_100'] = self.p90_95_99()[0]
                results['p95_100'] = self.p90_95_99()[1]
                results['p99_100'] = self.p90_95_99()[2]
            except Exception as e:
                results['p90_100'] = np.nan
                results['p95_100'] = np.nan
                results['p99_100'] = np.nan
        
            try:
                results['p99'] = self.p99()
            except Exception as e:
                results['p99'] = np.nan
        
            try:
                results['n05'] = self.n05()
            except Exception as e:
                results['n05'] = np.nan
        
            try:
                results['mad'] = self.mad()
            except Exception as e:
                results['mad'] = np.nan
        
            try:
                results['psi_sigvar'] = self.psi_sigvar()
            except Exception as e:
                results['psi_sigvar'] = np.nan
            try:
                results['kurtosis'] = self.Kurtosis()
            except Exception as e:
                results['kurtosis'] = np.nan
            try:
                results['ptp_amp'] = self.ptp_amplitude()
            except Exception as e:
                results['ptp_amp'] = np.nan
            try:
                results['rms'] = self.rms()
            except Exception as e:
                results['rms'] = np.nan
            try:
                results['rms_over_ptp_amp'] = self.rms_over_ptp_amp()
            except Exception as e:
                results['rms_over_ptp_amp'] = np.nan
        return results
