cd ..
set -e
# FFT

# python wear_main_loso_input_pruning.py --sparsity_weight_bin 0.1  --run_name 'InputPruningFineTuned' --preprocessing 'fft' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --log_name "FFT-0.1-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_fft.txt'

# python wear_main_loso_input_pruning.py --sparsity_weight_bin 2.0  --run_name 'InputPruningFineTuned' --preprocessing 'fft' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --log_name "FFT-2.0-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_fft.txt'
# # DCT
# python wear_main_loso_input_pruning.py --sparsity_weight_bin 2.0  --run_name 'InputPruningFineTuned' --preprocessing 'dct' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/dct/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'dct' --tflite-output-path 'models/tflite' --log_name "DCT-2.0-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_dct.txt'

# python wear_main_loso_input_pruning.py --sparsity_weight_bin 0.1  --run_name 'InputPruningFineTuned' --preprocessing 'dct' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/dct/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'dct' --tflite-output-path 'models/tflite' --log_name "DCT-0.1-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_dct.txt'

# # IHW
# python wear_main_loso_input_pruning.py --sparsity_weight_bin 0.4  --run_name 'InputPruningFineTuned' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/ihw/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --log_name "IHW-0.4-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_ihw.txt'

# python wear_main_loso_input_pruning.py --sparsity_weight_bin 1.0  --run_name 'InputPruningFineTuned' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/wear/stage1/ihw/wear_best_model_subject{subject}_val.pth' --performance --wandb False
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --log_name "IHW-1.0-quant" --wandb False --mask-log-file 'log/wear_loso_three_stage_results_ihw.txt'

# Channel pruning + Quant
for sparsity in 2.0; do
	python wear_main_loso_channel_pruning.py --preprocessing 'fft' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight "$sparsity" --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --wandb False
	python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'fft' --tflite-output-path "models/tflite/fft/${sparsity}" --log_name "CP-FFT-${sparsity}-quant" --wandb False
done

# for sparsity in 0.2 2.0; do
# 	python wear_main_loso_channel_pruning.py --preprocessing 'dct' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight "$sparsity" --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/dct/wear_best_model_subject{subject}_val.pth" --wandb False
# 	python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'dct' --tflite-output-path "models/tflite/dct/${sparsity}" --log_name "CP-DCT-${sparsity}-quant" --wandb False
# done

# for sparsity in 0.2 2.0; do
# 	python wear_main_loso_channel_pruning.py --preprocessing 'ihw' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight "$sparsity" --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/ihw/wear_best_model_subject{subject}_val.pth" --wandb False
# 	python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'ihw' --tflite-output-path "models/tflite/ihw/${sparsity}" --log_name "CP-IHW-${sparsity}-quant" --wandb False
# done

# # Dual pruning + Quant
# python wear_main_loso_five_stage.py --preprocessing 'fft' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight_bin 0.1 --sparsity_weight_channel 0.2 --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/fft/wear_best_model_subject{subject}_val.pth" --wandb False 
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path "models/tflite/wear/dp/fft/stage3/" --log_name "DP-stage3-FFT-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_fft.txt' 
# python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'fft' --tflite-output-path "models/tflite/wear/dp/fft/stage5/" --log_name "DP-stage5-FFT-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_fft.txt' 

# python wear_main_loso_five_stage.py --preprocessing 'dct' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight_bin 0.1 --sparsity_weight_channel 0.2 --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/dct/wear_best_model_subject{subject}_val.pth" --wandb False 
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'dct' --tflite-output-path "models/tflite/wear/dp/dct/stage3/" --log_name "DP-stage3-DCT-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_dct.txt' 
# python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'dct' --tflite-output-path "models/tflite/wear/dp/dct/stage5/" --log_name "DP-stage5-DCT-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_dct.txt' 

# python wear_main_loso_five_stage.py --preprocessing 'ihw' --run_name "Channel-pruning-wCEL-ForQuant" --sparsity_weight_bin 0.4 --sparsity_weight_channel 0.2 --performance --stage1_model_path "/home/qphan/master-thesis/models/wear/stage1/ihw/wear_best_model_subject{subject}_val.pth" --wandb False 
# python wear_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path "models/tflite/wear/dp/ihw/stage3/" --log_name "DP-stage3-ihw-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_ihw.txt' 
# python wear_quantize_loso_tflite_ptq.py --stage 5 --preprocessing 'ihw' --tflite-output-path "models/tflite/wear/dp/ihw/stage5/" --log_name "DP-stage5-ihw-quant" --wandb False --mask-log-file 'log/wear_loso_five_stage_results_ihw.txt' 


