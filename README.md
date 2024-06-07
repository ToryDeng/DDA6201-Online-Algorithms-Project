# Course Project of DDA6201: Selected Topics in Data and Decision Analytics I
This repo contains all code used in my course project of [DDA6201: Selected Topics in Data and Decision Analytics I (Machine Learning - Online Algorithms)](https://www.cuhk.edu.cn/en/course/11386). 

# Program structure
- `metrics/`: training losses of different models.
- `notebooks/`
  - `covid19_subset.ipynb` and `processing.ipynb`: code for preprocessing the dataset
  - `visualization.ipynb`: code for visualizing the results
- `data.py`: code for loading the dataset
- `models`: model definitions
- `train_base.py`: training the vanilla transformer
- `train_sp.py`: the supervised pretraining step
- `train_rl.py`: the reinforcement learning step
- `utils_rl.py`: utils for the reinforcement learning step

# Dataset
The dataset is a subset of cells sampled from patients with COVID-19 and healthy controls: https://doi.org/10.1016/j.cell.2021.01.053

# Models
1. vanilla transformer
2. Skipformer+SP
3. Skipformer+SP+HRL
4. Skipformer+SP+HRL+HO

> SP: supervised pretraining \
> HRL: hybrid reinforcement learning \
> HO: hyperparameter optimization


> The network architecture of Skipformer is adapted from [SkipNet](https://openaccess.thecvf.com/content_ECCV_2018/html/Xin_Wang_SkipNet_Learning_Dynamic_ECCV_2018_paper.html)

