# Exploring-DeepSORT

Official implementation of *Exploring the Effectiveness of Appearance Descriptor in DeepSORT*

Update on 2022/04: Our work is accpeted by IJCNN'22. Citation info has been updated.

## Steps to Use

### Clone Repo & Install Dependencies

```bash
git clone https://github.com/MrZilinXiao/Exploring-DeepSORT
cd Exploring-DeepSORT
pip install -r requirements.txt
```

### Prepare ReID dataset

This project provides a tool `tool/MOT16crop.py` that turns MOT-like dataset to ReID dataset. Make sure dataset is cropped correctly in `MOT16Cropped` before any experiments.

### Start Genetic-SORT

```bash
python genetic_sort_exp.py
```

Genetic-SORT reads configuration from `config.yaml` by default.

### Start Multi-Exp

```
# python multiexp.py --help
usage: multiexp.py [-h] [--exp_dir EXP_DIR] [--gpu GPU [GPU ...]] [--mot_eval_interval MOT_EVAL_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --exp_dir EXP_DIR
  --gpu GPU [GPU ...]
  --mot_eval_interval MOT_EVAL_INTERVAL
```

You may designate multiple GPUs by `python multiexp.py --gpu 0 1 2 3` to speed up training.

`multiexp.py` will read every `.yml` file in `experiments` if `EXP_DIR` is left default, and each YAML file corresponds to a experiment. An example experimental YAML file is like:

```yaml
dataset: MOT16
backbone: resnet50
pretrained: false
input_size: !!python/tuple [256, 128]
loss:
  type: softmax  # available options: [softmax, triplet, softmax_triplet, softmax_center, softmax_triplet_center]
  margin: 0.3  # valid only when using triplet loss
  center_weight: 0.005 # valid only when using center loss
  label_smoothing:
    enable: false
    epsilon: 0.1

sampler: softmax  # available options: [softmax, triplet]

neck_feat: after
bnn_neck: false
warmup:
  enable: false
  method: linear
  steps: !!python/tuple [10, 20]
  factor: 1.0 / 3
  max_epoch: 30

last_stride: 2

general:
  batch_size: 128
  num_workers: 4
  max_epoch: 30
  optim: Adam
  lr: 3.5e-4
  bias_lr_factor: 2
  weight_decay: 5e-4
  bias_weight_decay: 0.0
  center_lr: 0.5
  momentum: 0.9
transforms:
  random_horizontal_flip_p: 0.5
  padding: 10
  random_erasing: false
  random_erasing_p: 0.5

```



## Citation

```
@INPROCEEDINGS{9892052,
  author={Xiao, Zilin and Sun, Yanan},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Exploring the Effectiveness of Appearance Descriptor in DeepSORT}, 
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN55064.2022.9892052}
}
```



