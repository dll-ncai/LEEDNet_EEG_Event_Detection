import torch
import numpy as np
import pandas as pd
import os
import pywt
from scipy.signal import bessel, lfilter, stft
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def bessel_bandpass(data, lowcut=0.01, highcut=15, fs=200, order=4):
    b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
    return lfilter(b, a, data)

class EEGDataset(Dataset):
    """
    Unified Dataset class that smartly handles:
      - 1D branches (Raw, DWTs)
      - 2D branches (FFT, CWTs)
      - Combinations of Active Branches
      - Missing precomputed .npy files (falls back to calculating from .npz)
    """
    def __init__(self, csv_path, active_branches, wavelet_type='cmor', num_channels=1):
        self.num_channels = num_channels
        self.active_branches = active_branches
        self.wavelet_type = wavelet_type
        
        df = pd.read_csv(csv_path)
        self.file_list = df['file_name'].tolist()
        self.labels = df['abnormality_type_3_class'].tolist()

    def __len__(self):
        return len(self.file_list)

    def get_all_labels(self):
        return self.labels

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # Base zero branches
        x1 = torch.zeros((1, 1, 400), dtype=torch.float32)
        x2 = torch.zeros((1, 33, 14), dtype=torch.float32)
        x3 = torch.zeros((1, 30, 400), dtype=torch.float32)
        
        is_npy = file_path.endswith('.npy')
        
        if is_npy:
            feat = np.load(file_path).astype(np.float32)
            
            # --- Load Precomputed Branch ---
            # If it's a 1D pure scalar array (e.g. Symlet DWT, MSPCA)
            if feat.ndim == 1:
                x1 = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            elif feat.ndim == 2 and feat.shape[0] < 30: 
                # The 1D model branch expects [Batch, 1, Features].
                # VMD gives us [5 modes, 400], so we flatten it to [1, 1, 2000] 
                feat_flat = feat.flatten()
                x1 = torch.tensor(feat_flat, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            elif feat.ndim == 2 and feat.shape[0] == 33 and feat.shape[1] == 14:
                # Precomputed STFT/FFT spectrum [33, 14]
                x2 = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            # If it's 2D CWT [30, 400], it belongs to Branch 2
            elif feat.ndim == 2 and feat.shape[0] == 30 and feat.shape[1] == 400:
                x3 = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                
            # --- Dynamic Fallback for missing combinatory branches ---
            if (0 in self.active_branches and x1.sum() == 0) or (1 in self.active_branches and x2.sum() == 0):
                npz_name = os.path.basename(file_path).replace('.npy', '.npz')
                
                # 1. Try to load precomputed FFT if we need it
                if 1 in self.active_branches and x2.sum() == 0:
                    # Infer split from file_path
                    split = 'train' if 'train' in file_path else ('eval' if 'eval' in file_path else 'test')
                    fft_path = os.path.join('/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/preprocessed/exp9', split, npz_name.replace('.npz', '.npy'))
                    if os.path.exists(fft_path):
                        fft_feat = np.load(fft_path).astype(np.float32)
                        if fft_feat.ndim == 2 and fft_feat.shape[0] == 33 and fft_feat.shape[1] == 14:
                            x2 = torch.tensor(fft_feat, dtype=torch.float32).unsqueeze(0)
                
                # 2. If we still need raw data (for branch 0 or if FFT was missing)
                if (0 in self.active_branches and x1.sum() == 0) or (1 in self.active_branches and x2.sum() == 0):
                    base_root = '/media/hira/SSD 4TB/EEG Datasets/experiment/nouman_conv_2/'
                    
                    found_npz = ""
                    for s in ['eval', 'train', 'test']:
                        attempt = os.path.join(base_root, s, npz_name)
                        if os.path.exists(attempt):
                            found_npz = attempt
                            break
                            
                    if found_npz:
                        eeg_raw = np.load(found_npz)['data'][0:self.num_channels]
                        filtered = bessel_bandpass(eeg_raw)
                        if 0 in self.active_branches and x1.sum() == 0:
                            x1 = torch.tensor(filtered, dtype=torch.float32).unsqueeze(0)
                        if 1 in self.active_branches and x2.sum() == 0:
                            spec_list = [np.abs(stft(c, fs=200, nperseg=64, noverlap=32)[2])**2 for c in filtered]
                            x2 = torch.tensor(np.mean(np.stack(spec_list, axis=0), axis=0), dtype=torch.float32).unsqueeze(0)

        else:
            # Baseline processing directly from .npz
            eeg_raw = np.load(file_path)['data'][0:self.num_channels]
            filtered = bessel_bandpass(eeg_raw)
            
            if 0 in self.active_branches:
                x1 = torch.tensor(filtered, dtype=torch.float32).unsqueeze(0)
            if 1 in self.active_branches:
                spec_list = [np.abs(stft(c, fs=200, nperseg=64, noverlap=32)[2])**2 for c in filtered]
                x2 = torch.tensor(np.mean(np.stack(spec_list, axis=0), axis=0), dtype=torch.float32).unsqueeze(0)
            if 2 in self.active_branches:
                coeffs, _ = pywt.cwt(eeg_raw, np.arange(1, 31), self.wavelet_type, sampling_period=1/200)
                if self.wavelet_type == 'cmor1.5-1.0':
                    x3 = torch.tensor(np.abs(coeffs), dtype=torch.float32).squeeze(1).unsqueeze(0)
                else:
                    x3 = torch.tensor(coeffs, dtype=torch.float32).squeeze(1).unsqueeze(0)
                
        # Shape correction
        if x1.dim() == 2:
           x1 = x1.unsqueeze(0)  
            
        return file_path, x1, x2, x3, torch.tensor(self.labels[idx], dtype=torch.float32)

def get_dataloaders(csv_suffix="", active_branches=[0], wavelet_type='cmor1.5-1.0', batch_size=None, eval_batch=None, num_channels=None):
    from src.config import get_csv_paths, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, NUM_CHANNELS
    
    batch_size = batch_size or TRAIN_BATCH_SIZE
    eval_batch = eval_batch or EVAL_BATCH_SIZE
    num_channels = num_channels or NUM_CHANNELS

    train_csv, val_csv, test_csv = get_csv_paths(csv_suffix)

    train_dataset = EEGDataset(train_csv, active_branches, wavelet_type, num_channels)
    val_dataset   = EEGDataset(val_csv,   active_branches, wavelet_type, num_channels)
    test_dataset  = EEGDataset(test_csv,  active_branches, wavelet_type, num_channels)

    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,  batch_size=eval_batch, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False)
    
    return train_loader, val_loader, test_loader
