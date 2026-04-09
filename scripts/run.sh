cd .. 
# FFT-Channel pruning
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.1-wCEL" --sparsity_weight 0.1 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.2-wCEL" --sparsity_weight 0.2 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.3-wCEL" --sparsity_weight 0.3 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.4-wCEL" --sparsity_weight 0.4 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.5-wCEL" --sparsity_weight 0.5 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"

# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.6-wCEL" --sparsity_weight 0.6 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.7-wCEL" --sparsity_weight 0.7 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning0.8-wCEL" --sparsity_weight 0.8 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"

# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning1.0-wCEL" --sparsity_weight 1 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning2.0-wCEL" --sparsity_weight 2 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --run_name "Channel-pruning4.0-wCEL" --sparsity_weight 4 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# FFT-Input pruning
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.05 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft' 
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.1 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.2 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.3 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.4 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.5 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.6 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.7 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 0.8 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft' 
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 1.0 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'
python wear_main_loso_input_pruning.py --run_name "Input-pruning-Fine-tuned" --sparsity_weight_bin 2.0 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --preprocessing 'fft'



# IHM-Channel pruning
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.05-wCEL" --sparsity_weight 0.05 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.1-wCEL" --sparsity_weight 0.1 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.2-wCEL" --sparsity_weight 0.2 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.3-wCEL" --sparsity_weight 0.3 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.4-wCEL" --sparsity_weight 0.4 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.5-wCEL" --sparsity_weight 0.5 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.6-wCEL" --sparsity_weight 0.6 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning0.8-wCEL" --sparsity_weight 0.8 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning1.0-wCEL" --sparsity_weight 1 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "IHW-Channel-pruning2.0-wCEL" --sparsity_weight 2 --performance --stage1_model_path "/root/master-thesis/models/stage1/ihw/wear_best_model_subject{subject}_val.pth"

# FFT-Random pruning
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.05 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.1 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.2 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.3 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.4 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.5 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.6 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.7 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.8 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_random.py --preprocessing 'fft' --pruning_ratio 0.9 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"

# FFT-L1-norm pruning
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.05 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.1 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.2 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.3 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.4 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.5 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.6 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.7 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.8 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"
# python wear_main_loso_channel_pruning_static.py --preprocessing 'fft' --pruning_ratio 0.9 --performance --stage1_model_path "/root/master-thesis/models/stage1/fft/wear_best_model_subject{subject}_val.pth"

# Final: FFT-Dual-stage, bin=0.15, channels=0.3
# python wear_main_loso_five_stage.py --preprocessing 'fft' --run_name "FFT-Five-stage-bin0.15-channel0.3" --sparsity_weight_bin 0.15 --sparsity_weight_channel 0.3 --performance --stage1_model_path "/root/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth"