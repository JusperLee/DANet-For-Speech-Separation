# DANet-For-Speech-Separation
Pytorch implement of DANet For Speech Separation

> Chen Z, Luo Y, Mesgarani N. Deep attractor network for single-microphone speaker separation[C]//2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2017: 246-250.

## Requirement

- **Pytorch 0.4.0**
- **librosa 0.7.1**
- **PyYAML 5.1.2**

Due to the RNN multi-GPU parallel problem, only PyTorch 0.4.0 is supported.

## Training steps
1. First, you can use the create_scp script to generate training and test data scp files.

```shell
python create_scp.py
```

2. Then, in order to reduce the mismatch of training and test environments. Therefore, you need to run the util script to generate a feature normalization file (CMVN).

```shell
python ./utils/util.py
```

3. Finally, use the following command to train the network.

```shell
python train.py -opt ./option/train.yml
```
