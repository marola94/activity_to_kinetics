import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import zscore
from units import ureg
import warnings


def Michaelis_Menten(s, vmax, Km):
    return (vmax * s) / (Km + s)

def Substrate_Inhibition(s, vmax, Km, Ki):
    return (vmax*s)/(Km+s*(1+s/Ki))

def detect_outliers(residuals, threshold=2):
    z_scores = zscore(residuals)
    outliers = np.where(np.abs(z_scores) > threshold)[0]
    return outliers

def rule_based_outlier_detection(r):
    outliers = []
    for i in range(1, len(r) - 1):
        if (r[i] > r[i-1] and r[i] > r[i+1]) or (r[i] < r[i-1] and r[i] < r[i+1]):
            outliers.append(i)
    return np.array(outliers)

def detect_outliers_mad(residuals, threshold=3.5):
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    lower_bound = median_residual - threshold * mad
    upper_bound = median_residual + threshold * mad
    outliers = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
    return outliers

def combine_outlier_methods(V, residuals, mad_threshold=3.5):
    rule_based_outliers = rule_based_outlier_detection(V)
    mad_outliers = detect_outliers_mad(residuals, mad_threshold)
    combined_outliers = np.unique(np.concatenate((rule_based_outliers, mad_outliers)))
    return combined_outliers.astype(int)

def model_fitting(concentrations, rates, **kwargs):

    def _nonneg_rate_mask(rates,wells):
        """Mask for non-negative rates (uses magnitudes)."""
        return np.asarray(rates.magnitude) >= 0
    
    def max_outliers_removable(n_points: int, min_points: int = 4) -> int:
        """Max number of points removable while keeping at least 'min_points."""
        return max(0, n_points - min_points)
    
    def _q_drop_indices(qty, idx):
        """Drop indices from a pint.Quantity using a keep mask."""
        n = len(qty); idx = np.atleast_1d(np.asarray(idx, int))
        keep = np.ones(n, bool); keep[idx] = False
        return qty[keep]
    
    # --------------------------- Parameters ---------------------------
    min_points_required = 4

    # -------------------------- Initializing --------------------------

    model_type = kwargs.get('Model type')
    model_number = kwargs.get('Model No.')
    model_wells = kwargs.get('data')['Wells'].values

    # Streamline units
    concentration_unit = concentrations.units # Get the concentration unit
    time_unit = concentrations.units / rates.units # Get the time unit from the rate
    rate_unit = concentration_unit/time_unit # Define rate unit
    rates = rates.to(rate_unit)

    mask = _nonneg_rate_mask(rates, model_wells)

    # Filter negative rates
    c = concentrations[mask]
    r = rates[mask]

    # Warn user about filtered negative rates
    if len(mask)>0:
        invalid_rates = model_wells[np.asarray(rates.magnitude) <= 0]
        if len(invalid_rates) > 1:
            values_str = [str(v) for v in invalid_rates]
            output = ", ".join(values_str[:-1]) + " and " + values_str[-1]
        else:
            values_str = [str(v) for v in invalid_rates]
            output = values_str
        print(f"\u2757 {len(invalid_rates)} rates from well {output} are negative and filtered from the reamining in the model fitting.\n")
              
        # Check for minimum required datapoints for kinetic modelling
        if not (len(r) >= min_points_required or len(c) == len(r)):
            print(f"Less than {min_points_required} rates are available to create a kinetic model.\n"\
                  f"Kinetics will not be computed for model {model_number}.")
            
            return {} # Return no kinetics for this model number
    
    # ---------------- Initial fit ----------------
    initial_model, model_type, Vmax, Km, kcat, r_squared, rmse, mae = safe_fit(c, r, **kwargs)

    models = []
    #models = store_best_models(models, initial_model)

    # Lock key naming to INITIAL model type
    if model_type == "Michaelis-Menten":
        model_acronym = "MM"
    elif model_type == "Substrate Inhibition":
        model_acronym = "SI"
    
        # ---------------- combinatorial removal (keep >= 4 points) ----------------
    N = len(r)
    max_rm = max_outliers_removable(N, min_points=min_points_required)

    if max_rm >= 1:
        # single point removals
        for i in range(N):
            c_new = _q_drop_indices(c, i)
            r_new = _q_drop_indices(r, i)
            if len(r_new) < min_points_required:
                continue
            kwargs['Removed indices'] = [int(i)]
            model_dict, model_type2, *_ = safe_fit(c_new, r_new, **kwargs)
            models = store_best_models(models, model_dict)

        # combinations of size 2..max_rm
        if max_rm >= 2:
            idx_all = np.arange(N, dtype=int)
            for k in range(2, max_rm + 1):
                for combo in combinations(idx_all, k):
                    c_new = _q_drop_indices(c, combo)
                    r_new = _q_drop_indices(r, combo)
                    if len(r_new) < min_points_required:
                        continue
                    kwargs['Removed indices'] = list(map(int, combo))
                    model_dict, model_type2, *_ = safe_fit(c_new, r_new, **kwargs)
                    models = store_best_models(models, model_dict)

    # ---------------- package results (keys locked to initial type) ----------------
    models_dict = {
        f'all_data_model_{model_acronym}': initial_model,
        f'1_model_{model_acronym}': models[0] if len(models) > 0 else None,
        f'2_model_{model_acronym}': models[1] if len(models) > 1 else None,
        f'3_model_{model_acronym}': models[2] if len(models) > 2 else None,
    }
    return models_dict
    

def store_best_models(models, new_model, max_models=5):
    # Append new model to collection of models
    models.append(new_model)
    # Sorting models first according to lowest RMSE, then (if two model are the same RMSE) number of datapoints included in the model  
    models.sort(key=lambda x: (x['RMSE'],-len(x['rates'])), reverse=False)
    return models[:max_models] # return the five highest ranked models

def MM_fit_model(s, r, enzyme_concentration):

    # Initial guess on kcat (maximum rate) and Km (half of maximum rate)
    initial_guess_mm = np.array([np.max(r).magnitude, s[np.argmin(np.abs(r - (0.5 * np.max(r))))].magnitude])
    bounds = ([0,0],[np.inf, np.inf])
    
    popt, pcov = curve_fit(Michaelis_Menten, s.magnitude, r.magnitude, maxfev=10000, p0=initial_guess_mm, bounds=bounds)
    # Retrive parameters
    Vmax, Km = popt
    Vmax = ureg.Quantity(Vmax, str(r.units))
    Km = ureg.Quantity(Km,str(s.units))
    kcat = Vmax/ureg.Quantity(enzyme_concentration.magnitude,enzyme_concentration.units)

    # Retrive standard deviation of parameters
    std = np.sqrt(np.diag(pcov))
    std_Vmax = ureg.Quantity(std[0], str(r.units))
    std_Km = ureg.Quantity(std[1], str(s.units))
    std_kcat = std_Vmax/ureg.Quantity(enzyme_concentration.magnitude,enzyme_concentration.units)

    r_fit = Michaelis_Menten(s.magnitude, *popt)
    residuals = r.magnitude - r_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((r.magnitude - np.mean(r.magnitude))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(r))
    mae = np.mean(np.abs(residuals))

    return Vmax, std_Vmax, Km, std_Km, kcat, std_kcat, residuals, r_squared, rmse, mae

def SI_fit_model(s, r, enzyme_concentration):

    # Initial guess on kcat (maximum rate) and Km (half of maximum rate)
    initial_guess_si = np.array([np.max(r).magnitude, s[np.argmin(np.abs(r - (0.5 * np.max(r))))].magnitude, 10])
    bounds = ([0,0,0],[np.inf, np.inf, np.inf])
    
    popt, pcov = curve_fit(Substrate_Inhibition, s.magnitude, r.magnitude, maxfev = 10000, p0=initial_guess_si, bounds=bounds)

    # Retrive parameters
    Vmax, Km, Ki = popt
    Vmax = ureg.Quantity(Vmax, str(r.units))
    Km = ureg.Quantity(Km,str(s.units))
    Ki = ureg.Quantity(Ki,str(s.units))
    kcat = Vmax/ureg.Quantity(enzyme_concentration.magnitude,enzyme_concentration.units)

    # Retrive standard deviation of parameters
    std = np.sqrt(np.diag(pcov))
    std_Vmax = ureg.Quantity(std[0], str(r.units))
    std_Km = ureg.Quantity(std[1], str(s.units))
    std_Ki = ureg.Quantity(std[2], str(s.units))
    std_kcat = std_Vmax/ureg.Quantity(enzyme_concentration.magnitude,enzyme_concentration.units)

    r_fit = Substrate_Inhibition(s.magnitude, Vmax.magnitude, Km.magnitude, Ki.magnitude)
    residuals = r.magnitude - r_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((r.magnitude - np.mean(r.magnitude))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / len(r))
    mae = np.mean(np.abs(residuals))

    return Vmax, std_Vmax, Km, std_Km, Ki, std_Ki, kcat, std_kcat, residuals, r_squared, rmse, mae


def get_best_models(mm_models, si_models, num_best=3):
    # Combine models from both dictionaries
    combined_models = []
    for key, model in mm_models.items():
        if model:
            combined_models.append((key, model, 'MM_models'))
    for key, model in si_models.items():
        if model:
            combined_models.append((key, model, 'SI_models'))
    
    # Sort the models based on R-squared values in descending order
    combined_models.sort(key=lambda x: x[1]['RMSE'], reverse=False)
    
    # Select the top models
    best_models = combined_models[:num_best]
    
    return best_models

def format_quantity(quantity,units):

    # Find the most appropriate unit
    best_unit = min(units, key=lambda unit: abs(quantity.to(unit).magnitude))
    converted_quantity = quantity.to(best_unit)
    return converted_quantity

def model_values(
    model_type, Vmax, std_Vmax, Km, std_Km, kcat, std_kcat,
    residuals, r_squared, rmse, mae,
    s, r,
    enzyme_conc,
    title=None, model_number=None,
    Ki=None, std_Ki=None,
    removed_indices=None
):
    model_dict = {
        'Model type': model_type,
        'Title': title,
        'Model No.': model_number,
        'Vmax': Vmax, 'std. Vmax': std_Vmax,
        'Km': Km, 'std. Km': std_Km,
        'kcat': kcat, 'std. kcat': std_kcat,
        'residuals': residuals,
        'r_squared': r_squared,
        'RMSE': rmse,
        'MAE': mae,
        's': s,
        'rates': r,
        'Enzyme concentration': enzyme_conc,
        'Removed indices': removed_indices
    }
    if model_type == 'Substrate Inhibition':
        model_dict.update({'Ki': Ki, 'std. Ki': std_Ki})
    return model_dict

def fitting(s, r, **kwargs):
    model_type   = kwargs.get('Model type')
    model_number = kwargs.get('Model No.')
    title        = kwargs.get('Title')
    enz_conc     = kwargs.get('Enzyme concentration')
    removed_idx  = kwargs.get('Removed indices', None)

    if model_type == 'Michaelis-Menten':
        Vmax, std_Vmax, Km, std_Km, kcat, std_kcat, residuals, r_squared, rmse, mae = MM_fit_model(s, r, enz_conc)
        model_dict = model_values(
            model_type=model_type,
            Vmax=Vmax, std_Vmax=std_Vmax,
            Km=Km, std_Km=std_Km,
            kcat=kcat, std_kcat=std_kcat,
            residuals=residuals, r_squared=r_squared, rmse=rmse, mae=mae,
            s=s, r=r,
            enzyme_conc=enz_conc,
            title=title, model_number=model_number,
            removed_indices=removed_idx
        )

    elif model_type == 'Substrate Inhibition':
        Vmax, std_Vmax, Km, std_Km, Ki, std_Ki, kcat, std_kcat, residuals, r_squared, rmse, mae = SI_fit_model(s, r, enz_conc)
        model_dict = model_values(
            model_type=model_type,
            Vmax=Vmax, std_Vmax=std_Vmax,
            Km=Km, std_Km=std_Km,
            kcat=kcat, std_kcat=std_kcat,
            residuals=residuals, r_squared=r_squared, rmse=rmse, mae=mae,
            s=s, r=r,
            enzyme_conc=enz_conc,
            title=title, model_number=model_number,
            Ki=Ki, std_Ki=std_Ki,
            removed_indices=removed_idx
        )

    return model_dict, model_type, Vmax, Km, kcat, r_squared, rmse, mae



# Optional: silence SciPy's OptimizeWarning like "Covariance could not be estimated"
warnings.simplefilter("ignore", OptimizeWarning)

def safe_fit(s, r, **kwargs):
    """Run fitting(...) and return its tuple, or None if the optimizer fails."""
    try:
        return fitting(s, r, **kwargs)
    except (RuntimeError, ValueError) as e:
        # RuntimeError: maxfev exceeded, etc.  ValueError: bad bounds / shapes
        print(f"Fit failed for {kwargs.get('Model No.')} (removed={kwargs.get('Removed indices')}): {e}")
        return None
