export CUDA_VISIBLE_DEVICES=0

for station_id in $(seq -w 0 9)
do
    for model_type in Transformer
    do
        python -u run.py \
            --is_training 1 \
            --root_path ./dataset/pv-data/PVOD_en \
            --data_path station0${station_id}.csv \
            --model_id PV${station_id}_128_64 \
            --model ${model_type} \
            --data PV \
            --freq t \
            --features MS \
            --task_name long_term_forecast \
            --seq_len 128 \
            --label_len 64 \
            --pred_len 64 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 14 \
            --dec_in 14 \
            --c_out 1 \
            --d_model 512 \
            --d_ff 2048 \
            --des 'Exp' \
            --itr 1
    done
done