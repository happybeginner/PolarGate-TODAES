# PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network
Official code repository for the paper: *PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network* (ACM Transactions on Design Automation of Electronic Systems, TODAES 2026).

If you plan to explore more potential downstream tasks, please feel free to discuss with us (Email: liujw@cse.cuhk.edu.hk). We are looking forward to collaborate with you!

## Abstract

Understanding the functionality of Boolean networks is crucial for processes such as functional equivalence checking, logic synthesis, and malicious logic identification. With the proliferation of deep learning in electronic design automation (EDA), graph neural networks (GNNs) are widely used to embed and-inverter graphs (AIGs)—a standard form of Boolean networks—into vectorized representations. A key challenge in applying GNNs for Boolean representation is that although GNNs can effectively encapsulate the structural properties of AIGs, they struggle to efficiently capture Boolean logic functionality. In this work, we focus on breaking this bottleneck by enhancing the functional representation capability of GNNs, proposing PolarGate, an efficient solution that not only aligns message passing with AIG logical functionality but also effectively integrates global information. Leveraging the intrinsic ambipolar states (0 and 1) of AIG nodes, PolarGate maps gate behavior into an ambipolar state space, customizes differentiable logical operators, and designs a functionality-aware message passing strategy. To further capture global circuit information, PolarGate integrates a structure-aware preprocessing module and a global linear attention module, transcending the locality constraint of message passing. Experimental results on two functionality-related basic tasks (signal probability prediction and truth-table distance prediction) and a downstream task (logic equivalence prediction) show that PolarGate outperform state-of-the-art GNN-based methods.

## Installation
```shell
conda create -n polargate2 python=3.10
conda activate polargate2
pip install -r requirements.txt
``` 

For detailed installation for [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention), please refer to [Installation](https://github.com/sustcsonglin/flash-linear-attention?tab=readme-ov-file#installation):
```shell
pip install flash-linear-attention
``` 

For [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d), we recommend to find the corresponding version of the package to install manually. For example, if you are using python 3.10, CUDA 11.8 and PyTorch 2.5.0, you can install the following versions:
```shell
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Directory Structure
```
AIGDataset
  ├── PolarGate_raw
  ├── PolarGate_processed
       ├── npz
           ├── pi_edges.npz
           ├── labels.npz
       ├── [circuit_name]
           ├── raw
               ├── node-feat.csv
               ├── prob.csv
               ├── signed_edge.csv
       ├── split
           ├── 0.01-0.01-0.98
           ├── 0.02-0.02-0.96
           ├── 0.05-0.05-0.9
           ├── 0.1-0.1-0.8
PolarGate2
  ├── layers.py
  ├── model.py
  ├── load_dataset.py
  ├── preprocess_data.py
  ├── train.py
  ├── run_0.sh
  ├── run_1.sh
  ├── metala

```

## Prepare Dataset
```bash
mkdir AIGDataset
cd AIGDataset
```
If you want a preprocessed dataset (ready-to-use), you can download it from here:
```bash
wget https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_processed.zip
unzip PolarGate_processed.zip 
```
If you want the raw dataset (in .bench format), you can download it from here. Then, you can use `preprocess_data.py` to generate data input supported by PolarGate.
```bash
wget https://github.com/BUPT-GAMMA/PolarGate/releases/download/dataset/PolarGate_raw.zip
unzip PolarGate_raw.zip 
```

## Model training
To train the model for signal probability prediction, use the following command:
```bash
python train.py --task 'prob' --model 'PolarGate_MetaLA' --device 0 --batch_size 256 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --feature_type 'one-hot' --in_dim 3
```

**Explanation:**
- `--task 'prob'`: Specifies the task as signal probability prediction.
- `--model 'PolarGate_MetaLA'`: Specifies the model name as PolarGate_MetaLA.
- `--device 0`: Indicates the GPU device number to use.
- `--batch_size 256`: Sets the batch size for training.
- `--split_file 0.05-0.05-0.9`: Indicates the data split proportions for training, validation, and testing.
- `--layer_num 9`: Sets the number of layers in the model.
- `--in_dim 3`: Sets the input dimension size.

Similarly, to train the model for truth-table distance prediction, use the following command:

```bash
python train.py --task 'tt' --model 'PolarGate' --device 0 --batch_size 256 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --in_dim 3
```

python train_multitask.py --task 'multi' --model 'PolarGate_MetaLA' --device 0 --batch_size 256 \
       --split_file 0.05-0.05-0.9 --layer_num 9 --in_dim 3
## Cite PolarGate
If PolarGate could help your project, please cite our work:
```bibtex
@article{liu2026polargate,
  title={PolarGate: Breaking the Functionality Representation Bottleneck of And-Inverter Graph Neural Network},
  author={Liu, Jiawei and Zhai, Jianwang and He, Xun and Zhao, Mingyu and Lin, Zhe and Shi, Chuan and Yu, Bei},
  journal={ACM Transactions on Design Automation of Electronic Systems},
  year={2026},
  publisher={ACM},
  doi={10.1145/3812548}
}
```
