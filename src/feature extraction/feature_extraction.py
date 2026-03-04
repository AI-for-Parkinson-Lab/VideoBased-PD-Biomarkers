import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from math import pi
import pickle
from scipy.stats import f_oneway, shapiro, ttest_ind, mannwhitneyu
from scipy.signal import butter, filtfilt
from scipy.signal import welch, correlate
from scipy.stats import entropy
import matplotlib.cm as cm
from scipy.stats import spearmanr
sns.set(style="whitegrid", palette="coolwarm", font_scale=1.2)


class feature_ext_analysis:
    def __init__(self, config):
        self.config = config
        
        
    def _extract_features(self, distances, fps):
         distances_ = np.array(distances)

     
         fs = 30.0                  
         order = 4                  
         nyq = 0.5 * fs             
         cutoff_frequencies =  9.0
         normal_cutoff = cutoff_frequencies / nyq  # normalize the cutoff
         b, a = butter(order, normal_cutoff, btype='low', analog=False)
         filtered_signal = filtfilt(b, a, distances_)
         distances_ = filtered_signal
         peaks, _ = find_peaks(distances_, distance=5 , height=np.mean(distances_)/2, prominence=np.mean(distances_)/2)
         troughs, _ = find_peaks(-distances_, distance=5, height=-np.mean(distances_), prominence=np.mean(distances_)/2)
                
         
            
         #peaks, troughs = self.remove_consecutive_events(distances_, peaks, troughs)
         ############################################################## Compute speed signal
         time_interval = 1 / fps
         speed_signal = np.diff(distances_) / time_interval # Speed = Δdistance / Δtime

         ###################################################  amp
        
         
         while len(peaks) > 0 and len(troughs) > 0 and troughs[0] > peaks[0]:
             peaks = peaks[1:]
         
         amplitudes = []
         amp_frame_numbers = []

         for i in range(min(len(peaks), len(troughs))):
             peak = peaks[i]
             valid_troughs = troughs[troughs < peak]
             if len(valid_troughs) == 0:
                 continue
             last_trough = valid_troughs[-1]
             amp = abs(distances_[peak] - distances_[last_trough])
             amplitudes.append(amp)
             amp_frame_numbers.append(peak)


         amp_frame_numbers = np.array(amp_frame_numbers).reshape(-1, 1)
         # Compute median and max amplitude
         median_amplitude = np.median(amplitudes)
         max_amplitude = np.max(amplitudes)
         avg_amplitude = np.mean(amplitudes)


         time_points_amp = np.arange(len(amplitudes)).reshape(-1, 1)
         model_amp = LinearRegression()
         model_amp.fit(time_points_amp, amplitudes)
         #model_amp.fit(amp_frame_numbers, amplitudes)

         amp_slope = model_amp.coef_[0]######################################################################################

         ################################################################# Generate per-cycle speed 
         per_cycle_speed_maxima = []
         per_cycle_speed_avg = []
         per_cycle_speed_avg_frame_numbers = []
         for i in range(len(amplitudes) - 1):
             start_idx = peaks[i]  # Start of the window
             end_idx = peaks[i + 1]  # End of the window
             window_speed = speed_signal[start_idx:end_idx]  # Slice the speed signal
             
             if len(window_speed) > 0:  # Ensure the window is not empty
                 per_cycle_speed_maxima.append(np.percentile(np.abs(window_speed), 95))
                 per_cycle_speed_avg.append(np.mean(np.abs(window_speed)))
                 # mid-frame to regress the average values (just for viz/trend)
                 avg_frame = (start_idx + end_idx) // 2
                 per_cycle_speed_avg_frame_numbers.append(avg_frame)
         
            # Compute the median and max of per-cycle speed maxima
         mean_percycle_max_speed = np.mean(per_cycle_speed_maxima)
         mean_percycle_avg_speed = np.mean(per_cycle_speed_avg)
         per_cycle_speed_avg_frame_numbers = np.array(per_cycle_speed_avg_frame_numbers).reshape(-1, 1)
         avg_speed = np.mean(np.abs(speed_signal))
         
         time_points_speed = np.arange(len(per_cycle_speed_avg)).reshape(-1, 1)
         model_speed = LinearRegression()
         model_speed.fit(time_points_speed, np.abs(per_cycle_speed_avg))
         #model_speed.fit(per_cycle_speed_avg_frame_numbers, np.abs(per_cycle_speed_avg))

         speed_slope = model_speed.coef_[0]#####################################################################################
         # Compute tapping intervals (time between consecutive maxima)
         tapping_intervals = np.diff(peaks) / fps

         median_tapping_interval = np.median(tapping_intervals)
         mean_tapping_interval = np.mean(tapping_intervals)

         time_points_ti = np.arange(len(tapping_intervals)).reshape(-1, 1)
         model_ti = LinearRegression()
         model_ti.fit(time_points_ti, tapping_intervals)
         ti_slope = model_ti.coef_[0]


         
         
         ################################## ratio decrement
         mid = len(amplitudes) // 2
         first_half = amplitudes[:mid]
         second_half = amplitudes[mid:]
         mean_first = np.mean(first_half)
         mean_second = np.mean(second_half)
         amp_decrement_ratio = mean_second / mean_first 
         mid = len(np.abs(speed_signal)) // 2
         first_half = np.abs(speed_signal)[:mid]
         second_half = np.abs(speed_signal)[mid:]
         mean_first = np.mean(first_half)
         mean_second = np.mean(second_half)
         speed_decrement_ratio = mean_second / mean_first 
         #############################  new amp ti decrement 
         

         
         

         
         ####################################################################   hesitation-halts

         std_tapping_intervals = np.std(tapping_intervals)
         cov_tapping_interval = std_tapping_intervals/mean_tapping_interval

         std_amp = np.std(amplitudes)
         cov_amp = std_amp/avg_amplitude

         std_per_cycle_speed_maxima = np.std(per_cycle_speed_maxima)
         cov_per_cycle_speed_maxima = std_amp/mean_percycle_max_speed

         std_per_cycle_speed_avg = np.std(per_cycle_speed_avg)
         cov_per_cycle_speed_avg = std_per_cycle_speed_avg/mean_percycle_avg_speed
         

         std_speed = np.std(speed_signal)
         cov_speed = std_speed/avg_speed



         # Compute total number of interruptions
         threshold = 2 * median_tapping_interval
         num_interruptions1 = sum(interval > threshold for interval in tapping_intervals)
         
         threshold = 1.5 * median_tapping_interval
         num_interruptions2 = sum(interval > threshold for interval in tapping_intervals)
         ################################################################################################

        
     
         features = {'avg_amplitude': avg_amplitude,
                     'avg_percycle_max_speed':mean_percycle_max_speed,
                     'avg_percycle_avg_speed':mean_percycle_avg_speed,
                     'avg_cycle_duration':mean_tapping_interval,
                     'amp_slope':amp_slope, 
                     'cycle_slope':ti_slope,
                     'speed_slope':speed_slope,
                     'cov_cycle_duration':cov_tapping_interval, 
                     'cov_amp':cov_amp, 
                     'cov_percycle_max_speed':cov_per_cycle_speed_maxima,
                     'cov_percycle_avg_speed':cov_per_cycle_speed_avg,
                     'num_interruptions':num_interruptions2,
                     }

         feat_name = ['ids', 'video_path', 'label'] + list(features.keys())
  
         
         return list(features.values()),  feat_name

    def load_data(self):
            with open(self.config['annotated_pkl_path'], 'rb') as f:
                annotated_data = pickle.load(f)
            
            
            combined_video_paths = annotated_data['video_path'] 
            combined_distances = annotated_data['distances'] 
            combined_keypoints = annotated_data['keypoints'] 
            combined_ids = annotated_data['id'] 
            combined_labels = annotated_data['label'] 
            combined_fps = annotated_data['fps'] 

            ############################################################# === Feature Extraction ===
            all_data = []
            
            for i in tqdm(range(len(combined_distances))):
                
                fps = combined_fps[i]
                combined_distance=combined_distances[i]
                
                
                video_length = len(combined_distance)                   
                if  video_length>= 4*fps:
                    print(i, video_length, combined_video_paths[i])
                    features,  feat_name = self._extract_features(combined_distance,fps)
                    row = [combined_ids[i], combined_video_paths[i], combined_labels[i]] + features
                    all_data.append(row)
                
                    
            df = pd.DataFrame(all_data, columns= feat_name)
            filename = f"combined_features_{self.config['video_cutting']}_cutoff.csv"
            self.csv_save_path = os.path.join(self.config['save_path'], filename)
            df.to_csv(self.csv_save_path, index=False)
     
        
if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))  # repo/src/feature_extraction
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))  # repo root

    CONFIG = {
        'annotated_pkl_path': os.path.join(project_root, "data/raw/video_keypoints.pkl"),
        'save_path': os.path.join(project_root, "data/processed"),
    }

    os.makedirs(CONFIG['save_path'], exist_ok=True)  
   
    feature_ext_ana = feature_ext_analysis(CONFIG)
    feature_ext_ana.load_data()
