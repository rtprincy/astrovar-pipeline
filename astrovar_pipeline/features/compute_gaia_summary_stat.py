import numpy as np
from scipy import stats

def calculate_variability_metrics(time, mag, mag_error, band_suffix='_g_fov'):
    """
    Calculates statistical parameters for a Gaia light curve, corresponding to the 
    fields in the Gaia DR3 vari_summary table.
    
    Parameters:
    -----------
    time : array-like
        Array of observation times (e.g., Barycentric Julian Date).
    mag : array-like
        Array of magnitudes (G, BP, or RP).
    mag_error : array-like
        Array of magnitude errors.
    band_suffix : str, optional
        Suffix to append to keys (default '_g_fov' for G-band). 
        Use '_bp' or '_rp' for other bands.

    Returns:
    --------
    dict
        Dictionary containing the statistical parameters with keys matching 
        Gaia DR3 vari_summary table columns.
    """
    
    # Ensure inputs are numpy arrays and sort by time
    time = np.array(time)
    mag = np.array(mag)
    mag_error = np.array(mag_error)
    
    # Remove NaNs/Infs if present
    mask = np.isfinite(time) & np.isfinite(mag) & np.isfinite(mag_error)
    time = time[mask]
    mag = mag[mask]
    mag_error = mag_error[mask]
    
    # Sort by time (crucial for Abbe and Time Duration)
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    mag = mag[sort_idx]
    mag_error = mag_error[sort_idx]
    
    n_obs = len(mag)
    
    if n_obs < 2:
        return {f"num_selected{band_suffix}": n_obs} # Not enough data for stats

    # --- 1. num_selected ---
    # Total number of observations selected for variability analysis
    num_selected = n_obs

    # --- 2. mean_obs_time ---
    # Mean observation time
    mean_obs_time = np.mean(time)

    # --- 3. time_duration ---
    # Time duration of the time series
    time_duration = np.max(time) - np.min(time)

    # --- 4. min_mag ---
    min_mag = np.min(mag)

    # --- 5. max_mag ---
    max_mag = np.max(mag)

    # --- 6. mean_mag ---
    mean_mag = np.mean(mag)

    # --- 7. median_mag ---
    median_mag = np.median(mag)

    # --- 8. range_mag ---
    # Difference between highest and lowest magnitudes
    range_mag = np.ptp(mag) # Peak to peak (max - min)

    # --- 9. trimmed_range_mag ---
    # Trimmed difference (usually 5th and 95th percentiles for Gaia)
    p5 = np.percentile(mag, 5)
    p95 = np.percentile(mag, 95)
    trimmed_range_mag = p95 - p5

    # --- 10. std_dev_mag ---
    # Square root of the unweighted magnitude variance (ddof=1 for sample std)
    std_dev_mag = np.std(mag, ddof=1)

    # --- 11. skewness_mag ---
    # Standardized unweighted skewness (Fisher-Pearson)
    skewness_mag = stats.skew(mag, bias=False)

    # --- 12. kurtosis_mag ---
    # Standardized unweighted kurtosis (Fisher excess kurtosis)
    kurtosis_mag = stats.kurtosis(mag, bias=False)

    # --- 13. mad_mag ---
    # Median Absolute Deviation (raw median of deviations)
    # Note: Gaia documentation implies 'mad' here is the median of absolute deviations from the median.
    # It does not typically apply the 1.4826 scaling factor in this specific field, 
    # whereas 'std_dev' uses the standard variance.
    median_val = np.median(mag)
    mad_mag = np.median(np.abs(mag - median_val))

    # --- 14. abbe_mag ---
    # Abbe value: mean square successive difference / (2 * variance)
    # A = (1 / (2*(N-1))) * sum((m_{i+1} - m_i)^2) / var
    diffs = np.diff(mag)
    sum_sq_diffs = np.sum(diffs**2)
    variance = np.var(mag, ddof=1)
    
    if variance > 0:
        abbe_mag = (sum_sq_diffs / (2 * (n_obs - 1))) / variance
    else:
        abbe_mag = np.nan

    # --- 15. iqr_mag ---
    # Interquartile range (75th - 25th percentile)
    q75, q25 = np.percentile(mag, [75 ,25])
    iqr_mag = q75 - q25

    # --- 16. stetson_mag ---
    # Stetson variability index (J).
    # Requires pairs of observations. For a single band, we pair observations 
    # that are very close in time (e.g., within 0.02 days/30 mins).
    # If no pairs are found, this metric may be ill-defined for single-transit data.
    stetson_mag = calculate_stetson_j(time, mag, mag_error)

    # --- 17. std_dev_over_rms_err_mag ---
    # Signal-to-Noise estimate: StdDev / Root Mean Square Error
    # RMS Error = sqrt(mean(error^2))
    rms_err = np.sqrt(np.mean(mag_error**2))
    if rms_err > 0:
        std_dev_over_rms_err_mag = std_dev_mag / rms_err
    else:
        std_dev_over_rms_err_mag = np.nan

    # --- 18. outlier_median_g_fov ---
    # Greatest absolute deviation from the median normalized by the error
    # max( |mag - median| / error )
    deviations = np.abs(mag - median_mag)
    normalized_devs = deviations / mag_error
    outlier_median = np.max(normalized_devs)

    # Construct Dictionary with dynamic keys
    results = {
        f"num_selected{band_suffix}": num_selected,
        f"mean_obs_time{band_suffix}": mean_obs_time,
        f"time_duration{band_suffix}": time_duration,
        f"min_mag{band_suffix}": min_mag,
        f"max_mag{band_suffix}": max_mag,
        f"mean_mag{band_suffix}": mean_mag,
        f"median_mag{band_suffix}": median_mag,
        f"range_mag{band_suffix}": range_mag,
        f"trimmed_range_mag{band_suffix}": trimmed_range_mag,
        f"std_dev_mag{band_suffix}": std_dev_mag,
        f"skewness_mag{band_suffix}": skewness_mag,
        f"kurtosis_mag{band_suffix}": kurtosis_mag,
        f"mad_mag{band_suffix}": mad_mag,
        f"abbe_mag{band_suffix}": abbe_mag,
        f"iqr_mag{band_suffix}": iqr_mag,
        f"stetson_mag{band_suffix}": stetson_mag,
        f"std_dev_over_rms_err_mag{band_suffix}": std_dev_over_rms_err_mag,
        f"outlier_median{band_suffix}": outlier_median
    }

    return results

def calculate_stetson_j(time, mag, mag_error, pair_window=0.02):
    """
    Calculates the Welch-Stetson J index for a single band by identifying 
    pairs of observations within a small time window.

    Parameters:
    -----------
    time : array-like (sorted)
    mag : array-like
    mag_error : array-like
    pair_window : float
        Time window to consider points as a "pair" (in days). 
        Gaia FOV transits are usually separated by weeks, but a single 
        transit might contain multiple CCDs. If this is transit data, 
        pairs might not exist.

    Returns:
    --------
    float : Stetson J index (0.0 if no pairs found)
    """
    n = len(time)
    if n < 2:
        return 0.0

    # Calculate residuals scaled by error
    mean_mag = np.mean(mag) # Or weighted mean
    delta = np.sqrt(n / (n - 1)) * (mag - mean_mag) / mag_error
    
    # Identify pairs
    P_k = []
    weights = []
    
    # Iterate through sorted time to find pairs i, j such that time[j] - time[i] < window
    # Simple strategy: compare i with i+1
    for i in range(n - 1):
        dt = time[i+1] - time[i]
        if dt < pair_window:
            # Found a pair
            p_val = delta[i] * delta[i+1]
            P_k.append(p_val)
            weights.append(1.0) # Uniform weighting for now
            
    if len(P_k) == 0:
        return 0.0 # No close pairs found
    
    # Stetson J = sum(w * sgn(P) * sqrt(|P|)) / sum(w)
    P_k = np.array(P_k)
    J = np.sum(np.sign(P_k) * np.sqrt(np.abs(P_k))) / len(P_k)
    return J

# --- Example Usage ---
if __name__ == "__main__":
    # Generate synthetic light curve data
    np.random.seed(42)
    n_points = 50
    
    # Time: Random observations over 1000 days
    t = np.sort(np.random.uniform(0, 1000, n_points))
    
    # Mag: Sinusoidal variation + noise
    true_mag = 15.0 + 0.5 * np.sin(2 * np.pi * t / 100.0)
    noise = np.random.normal(0, 0.02, n_points)
    m = true_mag + noise
    
    # Error: Constant error for simplicity
    err = np.full(n_points, 0.02)

    # Calculate statistics
    stats_g = calculate_variability_metrics(t, m, err, band_suffix='_g_fov')

    print("--- Gaia Light Curve Statistics (G-Band) ---")
    for key, value in stats_g.items():
        print(f"{key}: {value:.4f}")