# REEDNet: An Efficient Network for Event Detection from Raw EEG Signals

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Citation](#citation)

This repository contains the source code for 'REEDNet: An Efficient Network for Event Detection from Raw EEG Signals' using data from **"NUST-MH-TUKL EEG Events"**.

REEDNet is a compact 1D deep learning model designed to classify EEG events directly from raw signals with minimal preprocessing. Compared to large-scale vision backbones (GoogLeNet, VGG16), REEDNet achieves competitive diagnostic accuracy while requiring over 100× fewer FLOPs and parameters, making it well suited for real-time, resource-constrained clinical settings.

* Muhammad Ali Alqarni, Hira Masood, Hassan Aqeel Khan, Awais Mehmood Kamboh, Saima Shafait, Tooba Jaffary and Faisal Shafait (2025) * **"REEDNet: An Efficient Network for Event Detection from Raw EEG Signals",** IEEE Access. doi: ()

# Requirements
1. Install mne library available at https://github.com/mne-tools/mne-python
2. This code was programmed in Python 3.10 (might work for other versions also)
3. Dataset available at: https://dll.seecs.nust.edu.pk/downloads/

# Dataset
The NMT-Events dataset consists of labeled EEG data collected from the Military Hospital (MH), Rawalpindi, between 2018 and 2021. The recordings were sampled at 200 Hz with 19-channel EEG following the international 10–20 system. Labels for Normal, Slowing Waves, and Spike-and-Wave discharges are provided in CSV files with corresponding timestamps.
* Total Recordings: 962 Normal, 113 Abnormal EEGs
* Format: EDF files (signals), CSV files (annotations)

# Citation
This repo was used to generate the results for the following paper on Event Detection in EEG.
  
[//]: # (  > Citation: Alqarni MA, Masood H, Qureshi AJ, Alvi M, Arbab H, Khan HA, Kamboh AM, Shafait S, and Shafait F &#40;2024&#41; **NeuroAssist: Open-Source Automatic Event Detection in Scalp EEG** IEEE Access &#40;__&#41;)
  > Citation: M. Ali Alqarni et al., "REEDNet: An Efficient Network for Event Detection from Raw EEG Signals", in IEEE Access, vol. xx, pp. xxx, 2025, doi: xxx. 
**BibTex Reference:**
