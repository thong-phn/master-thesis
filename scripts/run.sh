cd .. 
python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.1-wCEL" --sparsity_weight 0.1 --performance --stage1_model_path "/home/qphan/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.2-wCEL" --sparsity_weight 0.2 --performance --stage1_model_path "/home/qphan/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.3-wCEL" --sparsity_weight 0.3 --performance --stage1_model_path "/home/qphan/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.4-wCEL" --sparsity_weight 0.4 --performance --stage1_model_path "/home/qphan/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.5-wCEL" --sparsity_weight 0.5 --performance --stage1_model_path "/home/qphan/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"

