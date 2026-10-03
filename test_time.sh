
models=(PolarGate PolarGate_MetaLA)
tasks=(prob tt)

for model in ${models[@]}; do
    for task in ${tasks[@]}; do

        python test_time.py --dataset PolarGate_processed --task_type $task \
        --lr 0.001 --weight_decay 0.0001 --model $model --seed 2024\
        --in_dim 3 --out_dim 128 --layer_num 9 --device 2 --batch_size 16\
        --split_file 0.05-0.05-0.9 \
        --gradient_accumulation_steps 1
    done
done