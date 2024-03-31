<h2 align="center">
  <b>GAMMA: Graspability-Aware Mobile MAnipulation Policy Learning based on Online Grasping Pose Fusion</b>

  <b><i>ICRA 2024</i></b>

<div align="center">
    <a href="https://https://2024.ieee-icra.org/" target="_blank">
    <img src="https://img.shields.io/badge/ICRA 2024-Conference paper-red"></a>
    <a href="https://arxiv.org/abs/2309.15459" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://pku-epic.github.io/GAMMA/" target="_blank">
    <img src="https://img.shields.io/badge/Page-GAMMA-blue" alt="Project Page"/></a>
</div>
</h2>

This is the official repository of [**GAMMA: Graspability-Aware Mobile MAnipulation Policy Learning based on Online Grasping Pose Fusion**](https://arxiv.org/abs/2309.15459).

For more information, please visit our [**project page**](https://pku-epic.github.io/GAMMA/).

## Installation
1. **Clone this repo**
   ```bash
   git clone https://github.com/user432/gamma.git
   ```

1. **Preparing conda env**
   ```bash
   # We require python>=3.9 and cmake>=3.14
   conda create -n gamma python=3.9 cmake=3.14.0
   conda activate gamma

   # install habitat-sim
   conda install habitat-sim=0.2.5 withbullet -c conda-forge -c aihabitat

   # install habitat-lab
   pip install -e habitat-lab

   # install habitat-baselines
   pip install -e habitat-baselines
   
   pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

   cd GLtreeAPI/
   pip install -e .

   cd ../graspness_implementation/graspnessAPI

   cd pointnet2_pytorch/pointnet2_ops_lib/
   pip install -e .

   cd ../../knn
   pip install -e .

   cd ../graspnetAPI
   pip install -e .

   cd ..
   pip install -e .
   ```
1. **Scene datasets**
    ```
    # ReplicaCAD dataset
    python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset

    # YCB dataset
    python -m habitat_sim.utils.datasets_download --uids ycb 
    ```


1. **GSNet pre-trained model**
  
    Download the model from this [**link**](https://drive.google.com/file/d/1F_6EDdht1kr7bZCcgt54ieaIgWuvfS14/view?usp=sharing) and place it in `graspness_implementation/data/`

1. **Training and Evaluation**
    ```
    # Training command
    python -u -m habitat_baselines.run \
       --config-name=rearrange/rearrange_w_grasping.yaml

    # Evaluate your trained model ("eval_ckpt_path_dir" must be changed in the config)
    python -u -m habitat_baselines.run \
       --config-name=rearrange/rearrange_w_grasping.yaml \
        habitat_baselines.evaluate=True
    ```

## Debugging an environment issue

Our vectorized environments are very fast, but they are not very verbose. When using `VectorEnv` some errors may be silenced, resulting in process hanging or multiprocessing errors that are hard to interpret. We recommend setting the environment variable `HABITAT_ENV_DEBUG` to 1 when debugging (`export HABITAT_ENV_DEBUG=1`) as this will use the slower, but more verbose `ThreadedVectorEnv` class. Do not forget to reset `HABITAT_ENV_DEBUG` (`unset HABITAT_ENV_DEBUG`) when you are done debugging since `VectorEnv` is much faster than `ThreadedVectorEnv`.

## Datasets

[Common task and episode datasets used with Habitat-Lab](DATASETS.md).

Please download the Rearrange Pick ReplicaCAD episode dataset and change its path accordingly in `habitat-baselines/habitat_baselines/config/rearrange/rearrange_w_grasping.yaml`

## Citation
If you find our work useful in your research, please consider citing:

```
@misc{zhang2023gamma,
    title={GAMMA: Graspability-Aware Mobile MAnipulation Policy Learning based on Online Grasping Pose Fusion},
    author={Jiazhao Zhang and Nandiraju Gireesh and Jilong Wang and Xiaomeng Fang and Chaoyi Xu and Weiguang Chen and Liu Dai and He Wang},
    year={2023},
    eprint={2309.15459},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Contact
If you have any questions, please open a github issue or contact us:

Jiazhao Zhang: zhngjizh@gmail.com

Gireesh Nandiraju: f20170720h@alumni.bits-pilani.ac.in

He Wang: hewang@pku.edu.cn
