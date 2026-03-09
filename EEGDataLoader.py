import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pywt
from scipy.signal import lfilter, bessel, stft

class EEGDataset(Dataset):
    def __init__(self, files_list, number_of_eeg_channels):
        self.csv_file_path = files_list
        self.file_list = []
        self.number_of_eeg_channels = number_of_eeg_channels
        # Read the CSV file into a DataFrame
        df = pd.read_csv(self.csv_file_path)
        df['file_name'] = df['file_name']#.str.replace("SSD 4TB", "SSD 4TB1", regex=False)

        self.file_list = df['file_name'].tolist()
        self.ab_label = df['abnormality_type_3_class'].tolist()

    def __len__(self):
        return len(self.file_list)

    def get_all_labels(self):
        return self.ab_label  # already parsed in __init__

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        if file_path.endswith('.npy'):
            # Load pre-processed data
            # Assuming the .npy file contains the x1 branch data (Raw/Filtered)
            feat = np.load(file_path).astype(np.float32)
            
            # Reshape based on what's expected
            # if it's [400], it should be [1, 1, 400] for LEEDNet
            if feat.ndim == 1:
                eeg_data = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
            elif feat.ndim == 2:
                # [modes, T] -> [1, modes, T]
                eeg_data = torch.from_numpy(feat).unsqueeze(0)
            else:
                eeg_data = torch.from_numpy(feat)
                
            # Dummy data for Branch 1 (FFT) and Branch 2 (Wavelet) if they are not pre-processed
            # In LEEDNetModel, usually only one branch is active in the experiments the user is running
            # or they use the full model but we only have pre-processed data for x1 right now.
            power_spectrum = torch.zeros((1, 33, 14), dtype=torch.float32)
            wavelet_transform = torch.zeros((1, 30, 400), dtype=torch.float32)
            
        else:
            # Original .npz loading
            npz_data = np.load(file_path)
            eeg_data_raw = npz_data['data'][0:self.number_of_eeg_channels]

            filtered = bassel_bandpass_filter(eeg_data_raw, 0.01, 15, 200, order=4)
            spec_list = []
            for channel_data in filtered:
                f, t, Zxx = stft(channel_data, fs=200, nperseg=64, noverlap=32)
                power = np.abs(Zxx)**2  # shape: [freq_bins, time_steps]
                spec_list.append(power)

            spec_array = np.stack(spec_list, axis=0)
            power_spectrum_np = np.mean(spec_array, axis=0)  # shape: [freq, time]
            
            coefficients_cmor, frequencies_cmor = pywt.cwt(eeg_data_raw, np.arange(1, 31), 'cmor1.5-1.0', sampling_period=1/200)

            eeg_data = torch.tensor(filtered, dtype=torch.float32).unsqueeze(0) # [1, C, T]
            power_spectrum = torch.tensor(power_spectrum_np, dtype=torch.float32).unsqueeze(0)  # [1, F, T]
            wavelet_transform = torch.tensor(coefficients_cmor, dtype=torch.float32).squeeze(1).unsqueeze(0)

        label = self.ab_label[idx]
        return file_path, eeg_data, power_spectrum, wavelet_transform, torch.tensor(label, dtype=torch.float32)

def bassel_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
        y = lfilter(b, a, data)
        return y