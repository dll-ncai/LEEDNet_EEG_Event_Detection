# # There should always be a 'train' and 'eval' folder directly
# # below these given folders
# # Folders should contain all normal and abnormal data files without duplications
# data_folders = [
#     '../../dataset/nmt_scalp_eeg_dataset/normal',
#     '../../dataset/nmt_scalp_eeg_dataset/abnormal']
# # data_folders = [
# #     '../../dataset/testing/normal',
# #     '../../dataset/testing/abnormal']

# n_recordings = None  # set to an integer, if you want to restrict the set size
# sensor_types = ["EEG"]
# n_chans = 21
# max_recording_mins = None # exclude larger recordings from training set
# sec_to_cut = 60  # cut away at start of each recording
# duration_recording_mins = 10  # how many minutes to use per recording
# test_recording_mins = 10
# max_abs_val = 800  # for clipping
# sampling_freq = 100
# divisor = 10  # divide signal by this
# test_on_eval = True  # test on evaluation set or on training set
# # in case of test on eval, n_folds and i_testfold determine
# # validation fold in training set for training until first stop
# n_folds = 10
# i_test_fold = 9
# shuffle = True
# model_name = 'shallow'#shallow/deep for DNN (deep terminal local 1)
# n_start_chans = 25
# n_chan_factor = 2  # relevant for deep model only
# input_time_length = 6000
# final_conv_length = 1
# model_constraint = 'defaultnorm'
# init_lr = 1e-3
# batch_size = 64
# max_epochs = 35 # until first stop, the continue train on train+valid
# cuda = False # False

# Tayyib's configuration
# There should always be a 'train' and 'eval' folder directly
# below these given folders
# Folders should contain all normal and abnormal data files without duplications

# channel_to_keep = "FP1"

channel_to_keep = "Fp1-A1"
drop_channels = False

# referenced_electrodes = True
data_folders = [
	'/home/dll-1/Desktop/eeg/datasets/A_EEG_Data_Files/Processed_New/original/Abnormal',
	'/home/dll-1/Desktop/eeg/datasets/A_EEG_Data_Files/Processed_New/original/Normal']
file_saving_path= "/home/dll-1/Desktop/eeg/datasets/A_EEG_Data_Files/Processed_New/dataset_npz/"

referenced_electrodes = False
# data_folders = [
# 	'/home/arooba/ssd/hira/eeg/Datasets/testing/normal',
# 	'/home/arooba/ssd/hira/eeg/Datasets/testing/abnormal']
# file_saving_path = "/home/arooba/ssd/hira/eeg/Datasets/dataset_pickle/2020/test/"

# data_folders = [
# 	'/home/arooba/ssd/hira/eeg/Datasets/nmt_dataset/dataset_balanced/normal',
# 	'/home/arooba/ssd/hira/eeg/Datasets/nmt_dataset/dataset_balanced/abnormal']
# file_saving_path = "/home/arooba/ssd/hira/eeg/Datasets/dataset_pickle/2019/"


# data_folders = [
# 	'/home/arooba/ssd/hira/eeg/Datasets/nmt_dataset/dataset_balanced/normal',
# 	'/home/arooba/ssd/hira/eeg/Datasets/nmt_dataset/dataset_balanced/abnormal']

# data_folders = [
# 	'/home/arooba/ssd/hira/eeg/Datasets/channel_localization/edfs_year_2019_2020/abnormal_channel/'+channel_to_keep,
# 	'/home/arooba/ssd/hira/eeg/Datasets/channel_localization/edfs_year_2019_2020/normal']


n_recordings = None # set to an integer, if you want to restrict the set size
sensor_types = ["EEG"]
n_chans = 21
max_recording_mins = None # exclude larger recordings from training set
sec_to_cut = 60  # cut away at start of each recording
duration_recording_mins = 10  # how many minutes to use per recording
test_recording_mins = 10
max_abs_val = 7000  # for clipping#=-================
sampling_freq = 200#=-================
divisor = 9  # divide signal by this
test_on_eval = True  # test on evaluation set or on training set
# in case of test on eval, n_folds and i_testfold determine
# validation fold in training set for training until first stop
n_folds = 18
i_test_fold = 9
shuffle = True
model_name = 'shallow'#shallow/deep for DNN (deep terminal local 1)#=-================
n_start_chans = 25
n_chan_factor = 2  # relevant for deep model only
input_time_length = 5500
final_conv_length = 1
model_constraint = 'defaultnorm'
init_lr = 1e-3 #=-================Don’t alter this, the best configuration=====
batch_size = 128 #=-================
max_epochs = 55 #until first stop, the continue train on train+valid
cuda = True
# max_recording_length = 23 #276000
max_recording_length = 18 #216000   10 #120000

label_dict = ['normal', 'abnormal']
ab_label_dict = ['Normal','Sharp Wave','Delta Slow Wave', 'Spike and Wave Discharge', 'Beta Wave', 'Theta Wave', 'Triphasic Wave', 'Low Voltage','Burst Suppression','Unknown']
