models=(PolarGate PolarGate_MetaLA)
save_dir=output/
device=2

model=PolarGate
split=0.05
load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_9_state_dict.pth
python fintune_lec.py \
    --model $model \
    --device $device \
    --load_pretrain_path $load_pretrain_path \
    --save-dir ${save_dir}/${model}_split_${split} \

# model=PolarGate
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \

model=PolarGate_MetaLA
split=0.05
load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_MetaLA_9_state_dict.pth
python fintune_lec.py \
    --model $model \
    --device $device \
    --load_pretrain_path $load_pretrain_path \
    --save-dir ${save_dir}/${model}_split_${split} \

# model=PolarGate_MetaLA
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_MetaLA_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \