#!/bin/bash

# split_files=("0.01-0.01-0.98" "0.02-0.02-0.96" "0.05-0.05-0.9" "0.1-0.1-0.8")
# layer_nums=(1 3 5 7 9)

# for split in "${split_files[@]}"; do
#     for layers in "${layer_nums[@]}"; do
#         echo "Running with split_file=${split}, layer_num=${layers}"
        
#         python train.py \
#             --task 'prob' \
#             --model 'PolarGate_MetaLA' \
#             --device 0 \
#             --batch_size 16 \
#             --split_file "${split}" \
#             --layer_num "${layers}"
        
#         echo "----------------------------------------------"
#     done
# done

python train.py --task 'prob' --model 'PolarGate_MetaLA' --device 0 --batch_size 16 --split_file 0.01-0.01-0.98 --layer_num 9 --in_dim 3

python train.py --task 'prob' --model 'PolarGate_MetaLA' --device 0 --batch_size 16 --split_file 0.1-0.1-0.8 --layer_num 9 --in_dim 3

python train.py --task 'prob' --model 'PolarGate_MetaLA' --device 0 --batch_size 16 --split_file 0.02-0.02-0.96 --layer_num 9 --in_dim 3

python train.py --task 'prob' --model 'PolarGate_MetaLA' --device 0 --batch_size 16 --split_file 0.05-0.05-0.9 --layer_num 9 --in_dim 3

