cd ..
# python uci_quantize_locso_tflite_ptq.py --stage 1 --preprocessing 'fft' --tflite-output-path 'models/tflite'
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --mask-log-file 'log/keep/uci_loso_three_stage_input_pruning_results_fft.txt'

python uci_main_loso_input_pruning.py --sparsity_weight_bin 2.0 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'fft' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/fft/uci_best_model_subject{subject}_val.pth' --performance --wandb False
python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_fft.txt' --log_name 'FFT-2.0-quant' 

python uci_main_loso_input_pruning.py --sparsity_weight_bin 0.6 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth' --performance --wandb False
python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_ihw.txt' --log_name 'IHW-0.6-quant'


python uci_main_loso_input_pruning.py --sparsity_weight_bin 0.9 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth' --performance --wandb False
python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_ihw.txt' --log_name 'IHW-0.9-quant'
