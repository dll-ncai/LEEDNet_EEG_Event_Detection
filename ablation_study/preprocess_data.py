import os
import sys
import numpy as np
import pandas as pd
import torch
import pywt
from scipy.signal import lfilter, bessel
from tqdm import tqdm
from vmdpy import VMD

# --- Pre-processing Functions ---

def bessel_bandpass(data, lowcut=0.01, highcut=15, fs=200, order=4):
    b, a = bessel(order, [lowcut, highcut], btype='band', fs=fs)
    return lfilter(b, a, data)

# MSPCA (copied from notebooks for consistency)
def _hankel_avg(matrix, orig_len):
    K, L = matrix.shape
    result = np.zeros(orig_len); counts = np.zeros(orig_len)
    for i in range(K):
        result[i:i+L] += matrix[i]
        counts[i:i+L] += 1
    return result / counts

def mspca_denoise(signal, wavelet='db4', level=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sig = np.array(signal, dtype=np.float64).flatten()
    n = len(sig)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    rec = np.zeros(n)
    for i, c in enumerate(coeffs):
        L = len(c)
        K = n - L + 1
        if K <= 0: 
            temp_coeffs = [np.zeros_like(cc) for cc in coeffs]
            temp_coeffs[i] = c
            rec += pywt.waverec(temp_coeffs, wavelet)[:n]
            continue
        H = np.zeros((K, L))
        for j in range(K): H[j] = sig[j:j+L]
        H_t = torch.tensor(H, dtype=torch.float32).to(device)
        U, S, V = torch.svd(H_t)
        S_np = S.cpu().numpy(); s_sq = S_np**2; var = np.cumsum(s_sq) / np.sum(s_sq)
        rank = np.searchsorted(var, 0.95) + 1
        S_th = torch.zeros_like(S); S_th[:rank] = S[:rank]
        H_rec = torch.mm(U, torch.mm(torch.diag(S_th), V.t())).cpu().numpy()
        rec += _hankel_avg(H_rec, n)
    return rec / len(coeffs)

def vmd_decompose(signal, n_modes=5):
    # n_modes Intrinsic Mode Functions (IMFs)
    # alpha=2000, tau=0, K=n_modes, DC=0, init=1, tol=1e-7
    u, u_hat, omega = VMD(signal, 2000, 0, n_modes, 0, 1, 1e-7)
    return u # [n_modes, T]

# --- Main Script ---

def preprocess_experiment(exp_id, csv_files, base_save_dir, device):
    print(f"\n>>> Processing Experiment {exp_id}...")
    num_skipped = 0
    num_processed = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        new_data = []
        save_dir = os.path.join(base_save_dir, f"exp{exp_id}", os.path.basename(csv_path).replace('.csv', ''))
        os.makedirs(save_dir, exist_ok=True)
        
        desc = f"Exp{exp_id} - {os.path.basename(csv_path)}"
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            file_path = row['file_name']
            
            # Save Path Check
            save_name = os.path.basename(file_path).replace('.npz', '.npy')
            save_path = os.path.join(save_dir, save_name)
            
            if os.path.exists(save_path):
                num_skipped += 1
                new_row = row.copy()
                new_row['file_name'] = save_path
                new_data.append(new_row)
                continue
            
            try:
                raw_data = np.load(file_path)['data'][0] # Channel 0
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
            # Apply Pre-processing
            if exp_id == 1: # Symlet DWT
                filtered = bessel_bandpass(raw_data)
                coeffs = pywt.wavedec(filtered, 'sym5', level=4)
                feat = np.concatenate(coeffs).astype(np.float32)
            elif exp_id == 2: # Daubechies DWT
                filtered = bessel_bandpass(raw_data)
                coeffs = pywt.wavedec(filtered, 'db4', level=4)
                feat = np.concatenate(coeffs).astype(np.float32)
            elif exp_id == 3: # MSPCA
                feat = mspca_denoise(raw_data, device=device).astype(np.float32)
            elif exp_id == 4: # VMD
                filtered = bessel_bandpass(raw_data)
                feat = vmd_decompose(filtered, n_modes=5).astype(np.float32)
            elif exp_id == 5: # Baseline (t-SNE)
                feat = bessel_bandpass(raw_data).astype(np.float32)
            elif exp_id == 6: # Mexican Hat CWT (mexh)
                filtered = bessel_bandpass(raw_data)
                coeffs, _ = pywt.cwt(filtered, np.arange(1, 31), 'mexh', sampling_period=1/200)
                feat = coeffs.astype(np.float32)
            elif exp_id == 7: # Real Morlet CWT (morl)
                filtered = bessel_bandpass(raw_data)
                coeffs, _ = pywt.cwt(filtered, np.arange(1, 31), 'morl', sampling_period=1/200)
                feat = coeffs.astype(np.float32)
            elif exp_id == 8: # Gaussian Derivative CWT (gaus5)
                filtered = bessel_bandpass(raw_data)
                coeffs, _ = pywt.cwt(filtered, np.arange(1, 31), 'gaus5', sampling_period=1/200)
                feat = coeffs.astype(np.float32)
            elif exp_id == 9: # STFT / FFT Branch
                from scipy.signal import stft
                filtered = bessel_bandpass(raw_data)
                # raw_data is 1D, so filtered is 1D
                spec = np.abs(stft(filtered, fs=200, nperseg=64, noverlap=32)[2])**2
                feat = spec.astype(np.float32)
            
            # Save
            np.save(save_path, feat)
            num_processed += 1
            
            # Record
            new_row = row.copy()
            new_row['file_name'] = save_path
            new_data.append(new_row)
        
        # Save new CSV
        new_df = pd.DataFrame(new_data)
        
        # Special case mapping to match old suffix conventions
        suffix_str = f'_exp{exp_id}'
        if exp_id == 9: 
            suffix_str = '_fft'
            
        new_csv_path = csv_path.replace('.csv', f'{suffix_str}.csv')
        new_df.to_csv(new_csv_path, index=False)
        print(f"Saved new catalog to {new_csv_path}")
    
    print(f"Experiment {exp_id} finished: Processed {num_processed}, Skipped {num_skipped}")

if __name__ == "__main__":
    csv_files = [
        '/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/train.csv',
        '/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/eval.csv',
        '/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/test.csv'
    ]
    base_save_dir = '/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/preprocessed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run pre-processing for all experiments (1 through 9)
    for i in range(1, 10):
        preprocess_experiment(i, csv_files, base_save_dir, device)
    
    print("\\nAll pre-processing complete!")
