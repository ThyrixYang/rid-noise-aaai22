This is code of RID-Noise.

## Reproduce RID-Noise Results

### Toy tasks

Please refer to the notebook *ridnoise.ipynb* to view experiments on three toy tasks.

### Benchmarks

We provide the source code to reproduce the performance of RID-Noise in the Table 1. of the paper,
including the implementation of all the three benchmark tasks.


To reproduce the results of the Kinematics task with x dependent noise, please run

```bash
python train_and_inference.py --method rid --task noisy_kine_x --device cuda:0
```

We provide a bash script to reproduce all the results, please execute

```bash
bash run.sh
```

Please cite our paper if you find this repository helpful in your work.

```
@inproceedings{rid_noise,
  author    = {Jia{-}Qi Yang and
               Ke{-}Bin Fan and
               Hao Ma and
               De{-}Chuan Zhan},
  title     = {RID-Noise: Towards Robust Inverse Design under Noisy Environments},
  booktitle = {Thirty-Sixth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2022},
  year      = {2022}
}
```