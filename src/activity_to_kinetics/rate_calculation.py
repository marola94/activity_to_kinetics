import numpy as np
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression



def determine_initial_rate(timestamps,x,**datapoint_interval):

    if datapoint_interval:
        slope, intercept, r_value, _, _= manual_slope_fitting(
                    timestamps, 
                    x, 
                    datapoint_interval,
                    )
        return {'Initial rate':slope, 'Intercept':intercept,'Yield':x[-1], 'R2':r_value**2}

    else:
        print('\n Determining intial rate.')
        data_points = len(x) # The number of datapoint to fit saturation model to
        start_dp = 0
        end_dp = data_points

        reaction_thrsld = 3.5

        polyorders = [1,2,3,4]
        min_window_length = int(data_points*0.014)
        smoothed_data_1, _, errors_1, _, _, _ = savitzky_golay_filter(x,min_window_length,polyorders)

        if smoothed_data_1[-1] >= reaction_thrsld:
                
            reaction = 1
        else:
            reaction = 0

        for j in range(0,data_points):
    
                if j==0:
                    min_window_length = 0
                    if reaction == 0:
                        min_window_length = int(data_points*0.07)
                        polyorders = [1]

                    elif errors_1 > 30 and reaction == 1:
                        min_window_length = int(data_points*0.50)
                        polyorders = [1,2]

                    elif errors_1 > 10 and reaction == 1:
                        min_window_length = int(data_points*0.25)
                        polyorders = [1,2]
                    
                    elif errors_1 <= 10 and reaction == 1:
                        min_window_length = int(data_points*0.014)
                        polyorders = [1,2]

                    iteration = 1
                    previous_error = errors_1
                    smoothed_data_2, snr_2, errors_2, rmse_2, mse_2, std_2= savitzky_golay_filter(smoothed_data_1, min_window_length,polyorders)

                else:             
                    if (100-(errors_2/previous_error)*100) < 0.5:
                        
                        if errors_2 > 50:
                            polyorders = [1,2]
                            if j == 1:
                                window_increase = int(data_points*0.1)
                            else:
                                window_increase = int(data_points*0.069)
                        elif errors_2 > 20:
                            polyorders = [1,2]
                            if j == 1:
                                window_increase = int(data_points*0.069)
                            else:
                                window_increase = int(data_points*0.049)
                        else:
                            polyorders = [1]
                            window_increase = int(data_points*0.014)

                    else:
                        polyorders = [1,2]
                        window_increase = int(data_points*0.014)

                    min_window_length += window_increase
                    if min_window_length > data_points:
                        break
                    
                    iteration += 1
                    previous_error = errors_2
                    smoothed_data_2, snr_2, errors_2, rmse_2, mse_2, std_2= savitzky_golay_filter(smoothed_data_2, min_window_length,polyorders)
                
                start_dp = 0
                end_dp = int(data_points*0.021)
                slope, intercept, r_value, p_value, std_err = linregress(timestamps[start_dp:end_dp], smoothed_data_2[start_dp:end_dp])

                if errors_2 < 60 and slope > -0.0005:

                    peaks, _, _, _ = detect_lumps(smoothed_data_2)
                    #,'Times:',seconds_to_datetime(t[peaks]))
                    #first_derriv =  savgol_filter(smoothed_data_2, 10, 1, deriv=1)
                    #print(first_derriv)
                    if len(peaks) > 0:
                        noise = 'Yes'
                        
                    else:
                        noise = 'No'              


                    if reaction == 0:
                        for k in range(0,data_points, int(data_points*0.1)):
                            start_dp = 0
                            min_data_points = int(data_points*0.069)
                            end_dp = min_data_points + k
                            if k > data_points:
                                break
                            a, b, r, p, std = linregress(timestamps[start_dp:end_dp], smoothed_data_2[start_dp:end_dp])
                            if k==0:
                                slope = a
                                intercept = b
                                r_value = r
                                std_err = std

                            else:
                                
                                if min([a, slope], key=lambda x: abs(x - 0)) == a: # determine which slope is closest to zero
                                    slope = a
                                    intercept = b
                                    r_value = r
                                    std_err = std

                    
                    elif reaction == 1:

                        start_dp = 0
                        end_dp = int(data_points*0.021)
                        slope, intercept, r_value, p_value, std_err = linregress(timestamps[start_dp:end_dp], smoothed_data_2[start_dp:end_dp])

                        
                    if slope > -0.0005:

                        break

                    else:
                        polyorders = [1,2]
                        if reaction == 0:
                            min_window_length = int(data_points*0.14)
                            start_dp = 0
                            end_dp = int(data_points*0.069)
                        else:
                            min_window_length = int(data_points*0.042)
                            start_dp = 0
                            end_dp = int(data_points*0.069)

                        
                        if min_window_length > data_points:
                            break

                        iteration += 1
                        previous_error = errors_2
                        smoothed_data_2, snr_2, errors_2, rmse_2, mse_2= savitzky_golay_filter(smoothed_data_2, min_window_length,polyorders)

                        slope, intercept, r_value, p_value, std_err = linregress(timestamps[start_dp:end_dp], smoothed_data_2[start_dp:end_dp])

                        break
        
        produced_yield = smoothed_data_2[-1]

        y_filter = smoothed_data_2

        return {'Filtered curve':y_filter, 'Yield':produced_yield, 'Initial rate':slope, 'Intercept':intercept, 'R2':r_value**2}

def manual_slope_fitting(x,y, datapoint_interval):
    print('\n Determining intial rates using the given data point interval.') 

    start_dp = int(datapoint_interval['Start datapoint'])
    end_dp = int(datapoint_interval['End datapoint'])
    slope, intercept, r_value, p_value, std_err = linregress(x[start_dp:end_dp], y[start_dp:end_dp])

    return slope,intercept, r_value, p_value, std_err

def savitzky_golay_filter(data, min_window_length, polyorders):
    number_of_datapoints = len(data)
    
    window_lenghts = [number_of_datapoints]
    while number_of_datapoints > min_window_length:
        number_of_datapoints /= 2
        window_lenghts.append(int(number_of_datapoints))

    for polyorder in polyorders:
        for window in window_lenghts:
            
            if polyorder < window:
                

                if polyorder == polyorders[0] and window == window_lenghts[0]:
                    
                    smoothed_data_1 = savgol_filter(data, window, polyorder)

                    errors_1, std_1 = measure_noise(smoothed_data_1, 10)
                    mse_1, rmse_1 = calculate_residual_errors(data, smoothed_data_1)
                    snr_1 = calculate_snr(smoothed_data_1)
                    
                else:
                    smoothed_data_2 = savgol_filter(data, window, polyorder)

                    errors_2, std_2 = measure_noise(smoothed_data_2, 10)
                    mse_2, rmse_2 = calculate_residual_errors(data, smoothed_data_2)
                    snr_2 = calculate_snr(smoothed_data_2)


                    if rmse_1  >  rmse_2:
                        smoothed_data_1 = smoothed_data_2
                        snr_1 = snr_2
                        errors_1 = errors_2
                        rmse_1 = rmse_2
                        mse_1 =mse_2
                        std_1 = std_2

    # RMSE from residual_errors over the first 100 datapoints works best for curves with little noise, while error from measure_noise is better for curves with a lot of noise
    # OBJECTIVE: Find a way to detect curve with a lot of noise and then smooth them out -> this should be done iteratively by increasing the minimum window size in the Savitzky-Golay 
    # filter -> make a algorithm that keep increaseing the window size by 10 if the .... 

    return smoothed_data_1, snr_1, errors_1, rmse_1, mse_1, std_1

def detect_lumps(data):
    # Find the peaks in the slopes
    peak_indices, peak_properties = find_peaks(data, prominence=0.08)
    #prominences =  peak_prominences(data, peak_indices)
    widths, peak_heights, peak_beginnings, peak_ends = peak_widths(data, peak_indices,rel_height=1.0)#, prominence_data=prominences)
    #print('Widths:',widths)
    #print('Heights:', peak_heights)
    #print('Peak start:', peak_beginnings)
    #print('Peak ends:',peak_ends)
    return peak_indices+1, peak_heights, peak_beginnings, peak_ends  

def measure_noise(data, window_size):
    windows = sliding_window(data, window_size)
    rmse_list = []
    std_list = []
    r2_scores = []
    nonlinearity_scores = []
    for window in windows:
        X = np.arange(len(window)).reshape(-1, 1)
        y = window
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        rmse_list.append(rmse)
        
        std = np.std(y)
        std_list.append(std)
        #r2 = r2_score(y, y_pred)
        #r2_scores.append(r2)
        ssr = np.sum(np.square(y - y_pred))
        nonlinearity_scores.append(ssr)
    return np.sum(rmse_list), np.sum(std_list)

def calculate_residual_errors(original_data, filtered_data):
    residuals = original_data[0:int(len(original_data)*0.1)] - filtered_data[0:int(len(filtered_data)*0.1)]
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    return mse, rmse

def calculate_snr(data):
    signal_power = np.mean(data**2)
    noise_power = np.mean((data - np.mean(data))**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def sliding_window(data, window_size):
    num_windows = len(data) - window_size + 1
    windows = [data[i:i+window_size] for i in range(num_windows)]
    return windows