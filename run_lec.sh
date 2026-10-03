models=(PolarGate PolarGate_MetaLA)
save_dir=output
device=1
difficulties=(easy medium)
difficulty=easy
train_dir=/home/hx/polargate_extended/gnn_v4/gnn/dataset/train_${difficulty}

##############fintun_lec.py################
# model=PolarGate
# split=0.05
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \

# model=PolarGate
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \

# model=PolarGate_MetaLA
# split=0.05
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_MetaLA_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \

# model=PolarGate_MetaLA
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_MetaLA_9_state_dict.pth
# python fintune_lec.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${model}_split_${split} \




################fintun_lec2.py################
# model=PolarGate
# split=0.05
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_9_state_dict.pth
# python fintune_lec2.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${difficulty}/${model}_split_${split} \
#     --train-dir $train_dir

# model=PolarGate
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_9_state_dict.pth
# python fintune_lec2.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${difficulty}/${model}_split_${split} \
#     --train-dir $train_dir

model=PolarGate_MetaLA
split=0.05
load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_0.05-0.05-0.9_PolarGate_MetaLA_9_state_dict.pth
python fintune_lec2.py \
    --model $model \
    --device $device \
    --load_pretrain_path $load_pretrain_path \
    --save-dir ${save_dir}/${difficulty}/${model}_split_${split}_w_normalize \
    --train-dir $train_dir

# model=PolarGate_MetaLA
# split=1.0
# load_pretrain_path=/home/hx/polargate_extended/Polargate2/ft_saved/multi_PolarGate_processed_PolarGate_MetaLA_9_state_dict.pth
# python fintune_lec2.py \
#     --model $model \
#     --device $device \
#     --load_pretrain_path $load_pretrain_path \
#     --save-dir ${save_dir}/${difficulty}/${model}_split_${split} \
#     --train-dir $train_dir