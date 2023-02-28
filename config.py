# Root path to data loc
#root = "/Users/naokinonaka/dev_dir/data/"
root = "/home/nonaka/mnt"

# Physionet 2017
pnet_dirname = "physionet2017"
pnet_labels = ["N", "A", "O", "~"]

pnet_reference = "training2017/REFERENCE.csv"
pnet_max_duration = 61

# CPSC2018
cpsc_dirname = "CPSC2018"

cpsc_reference = "TrainingSet3/REFERENCE.csv"
cpsc_freq = 500 # From HP (http://2018.icbeb.org/Challenge.html)
cpsc_signal_denominator = 1
cpsc_max_duration = 80
cpsc_dxs = ["Normal", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]

# train set size
train_size = 0.6

# Directory
raw_dir = "raw"
processed_dir = "processed_augment_exp3"

# Experiment result saving settings
#result_dir = "./results"
result_dir = "/home/nonaka/mnt/results/ecg_augment/201130_results"
model_dir = f"{result_dir}/models/"
save_loc = f"{result_dir}/predictions/"
tflog_dir = f"{result_dir}/runs"

ep_str = "ep{:04d}"
modelfile = "{}-{}.pth"
train_logfile = "train_log.txt"
valid_logfile = "valid_log.txt"
