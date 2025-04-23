export CUDA_VISIBLE_DEVICES=1

for model_type in MLWT_symlet
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/pv-data/PVOD_en \
        --data_path stanford_2017to2019.csv \
        --model_id Stanford_${model_type}_128_64 \
        --model ${model_type} \
        --data Stanford \
        --task_name long_term_forecast \
        --freq t \
        --features MS \
        --seq_len 128 \
        --label_len 64 \
        --pred_len 64 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 20 \
        --dec_in 20 \
        --c_out 1 \
        --d_model 512 \
        --d_ff 2048 \
        --des 'Exp' \
        --itr 1
done