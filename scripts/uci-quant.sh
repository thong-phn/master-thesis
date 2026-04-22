cd ..
set -e
# python uci_quantize_locso_tflite_ptq.py --stage 1 --preprocessing 'fft' --tflite-output-path 'models/tflite'
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --mask-log-file 'log/keep/uci_loso_three_stage_input_pruning_results_fft.txt'

# python uci_main_loso_input_pruning.py --sparsity_weight_bin 2.0 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'fft' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/fft/uci_best_model_subject{subject}_val.pth' --performance --wandb False
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'fft' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_fft.txt' --log_name 'FFT-2.0-quant' 

# python uci_main_loso_input_pruning.py --sparsity_weight_bin 0.6 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth' --performance --wandb False
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_ihw.txt' --log_name 'IHW-0.6-quant'


# python uci_main_loso_input_pruning.py --sparsity_weight_bin 0.9 --run_name 'InputPruningFineTunedForQuant' --preprocessing 'ihw' --stage1_model_path '/home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth' --performance --wandb False
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing 'ihw' --tflite-output-path 'models/tflite' --mask-log-file 'log/uci_loso_three_stage_input_pruning_results_ihw.txt' --log_name 'IHW-0.9-quant'

# # Channel pruning + Quant
for sparsity in 4.0; do
    # python uci_main_loso_channel_pruning.py --preprocessing "no" --sparsity_weight ${sparsity} --performance --run_name "ChannelPruningFineTuned" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/no/uci_best_model_subject{subject}_val.pth --wandb 1
    python uci_quantize_loso_tflite_ptq.py --stage 7 --preprocessing "no" --tflite-output-path "models/tflite/uci/cp-no/${sparsity}" --log_name "CP-NO-${sparsity}-quant"
done
# for sparsity in 2.0; do
#     python uci_main_loso_channel_pruning.py --preprocessing "ihw" --sparsity_weight ${sparsity} --performance --run_name "ChannelPruningFineTuned" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth
#     python uci_quantize_loso_tflite_ptq.py --stage 7 --preprocessing "ihw" --tflite-output-path "models/tflite/uci/cp-ihw/${sparsity}" --log_name "CP-IHW-${sparsity}-quant"
# done

# for sparsity in 0.2 2.0; do
#     python uci_main_loso_channel_pruning.py --preprocessing "fft" --sparsity_weight ${sparsity} --performance --run_name "ChannelPruningFineTuned" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/fft/uci_best_model_subject{subject}_val.pth
#     python uci_quantize_loso_tflite_ptq.py --stage 7 --preprocessing "fft" --tflite-output-path "models/tflite/uci/cp-fft/${sparsity}" --log_name "CP-FFT-${sparsity}-quant"
# done

# for sparsity in 0.2 2.0; do
#     python uci_main_loso_channel_pruning.py --preprocessing "dct" --sparsity_weight ${sparsity} --performance --run_name "ChannelPruningFineTuned" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/dct/uci_best_model_subject{subject}_val.pth
#     python uci_quantize_loso_tflite_ptq.py --stage 7 --preprocessing "dct" --tflite-output-path "models/tflite/uci/cp-dct/${sparsity}" --log_name "CP-DCT-${sparsity}-quant"
# done

# # Dual pruning + Quant
# python uci_main_loso_five_stage.py --preprocessing "fft" --sparsity_weight_bin 0.1 --sparsity_weight_channel 0.2 --performance --run_name "FiveStage" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/fft/uci_best_model_subject{subject}_val.pth
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing "fft" --tflite-output-path "models/tflite/uci/dp/fft/stage3/" --log_name "DP-FFT-quant" --mask-log-file 'log/uci_loso_five_stage_results_fft.txt'
# python uci_quantize_loso_tflite_ptq.py --stage 5 --preprocessing "fft" --tflite-output-path "models/tflite/uci/dp/fft/stage5/" --log_name "DP-FFT-quant" --mask-log-file 'log/uci_loso_five_stage_results_fft.txt'

# python uci_main_loso_five_stage.py --preprocessing "dct" --sparsity_weight_bin 0.1 --sparsity_weight_channel 0.2 --performance --run_name "FiveStage" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/dct/uci_best_model_subject{subject}_val.pth
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing "dct" --tflite-output-path "models/tflite/uci/dp/dct/stage3/" --log_name "DP-DCT-quant" --mask-log-file 'log/uci_loso_five_stage_results_dct.txt'
# python uci_quantize_loso_tflite_ptq.py --stage 5 --preprocessing "dct" --tflite-output-path "models/tflite/uci/dp/dct/stage5/" --log_name "DP-DCT-quant" --mask-log-file 'log/uci_loso_five_stage_results_dct.txt'

# python uci_main_loso_five_stage.py --preprocessing "ihw" --sparsity_weight_bin 0.6 --sparsity_weight_channel 0.2 --performance --run_name "FiveStage" --stage1_model_path /home/qphan/master-thesis/models/uci/stage1/ihw/uci_best_model_subject{subject}_val.pth
# python uci_quantize_loso_tflite_ptq.py --stage 3 --preprocessing "ihw" --tflite-output-path "models/tflite/uci/dp/ihw/stage3/" --log_name "DP-IHW-quant" --mask-log-file 'log/uci_loso_five_stage_results_ihw.txt'
# python uci_quantize_loso_tflite_ptq.py --stage 5 --preprocessing "ihw" --tflite-output-path "models/tflite/uci/dp/ihw/stage5/" --log_name "DP-IHW-quant" --mask-log-file 'log/uci_loso_five_stage_results_ihw.txt'

# # Baseline + Quant
# python uci_quantize_loso_tflite_ptq.py --stage 1 --preprocessing "ihw" --tflite-output-path "models/tflite/uci/bl-ihw" --log_name "BL-IHW-quant"
# python uci_quantize_loso_tflite_ptq.py --stage 1 --preprocessing "dct" --tflite-output-path "models/tflite/uci/bl-fft" --log_name "BL-DCT-quant"
# python uci_quantize_loso_tflite_ptq.py --stage 1 --preprocessing "fft" --tflite-output-path "models/tflite/uci/bl-dct" --log_name "BL-FFT-quant"
# python uci_quantize_loso_tflite_ptq.py --stage 1 --preprocessing "no" --tflite-output-path "models/tflite/uci/bl-no" --log_name "BL-NO-quant"


