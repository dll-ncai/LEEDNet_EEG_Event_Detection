import os

# Base paths
BASE_DATASET_PATH = '/media/hira/SSD 4TB/code/LEEDNet_EEG_Event_Detection/ablation_study/dataset/'

# Model and Prediction Save Directories
MODEL_SAVE_DIR = '../dataset/model/weights/'
PRED_SAVE_DIR  = '../dataset/predictions/'

# Hyperparameters
TRAIN_BATCH_SIZE = 82
EVAL_BATCH_SIZE  = 82

# Audio/Signal specifics
NUM_CHANNELS = 1

def get_csv_paths(suffix=""):
    """
    Returns the mapped train/eval/test paths.
    Falls back to the base version if the suffixed one doesn't exist.
    """
    train_csv = os.path.join(BASE_DATASET_PATH, f'train{suffix}.csv')
    val_csv   = os.path.join(BASE_DATASET_PATH, f'eval{suffix}.csv')
    test_csv  = os.path.join(BASE_DATASET_PATH, f'test{suffix}.csv')
    
    if not os.path.exists(train_csv): train_csv = os.path.join(BASE_DATASET_PATH, 'train.csv')
    if not os.path.exists(val_csv):   val_csv   = os.path.join(BASE_DATASET_PATH, 'eval.csv')
    if not os.path.exists(test_csv):  test_csv  = os.path.join(BASE_DATASET_PATH, 'test.csv')
    
    return train_csv, val_csv, test_csv
