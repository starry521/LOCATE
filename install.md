# Installation of locate

- Python 3.7 + Cuda 11.6 + Pytorch 1.12.1

```
conda create -n locate -y python=3.7
conda activate locate
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

- Install other requirements
```
pip install -r requirements.txt
```