# TEDDCF
This is the pytorch implementation of TEDDCF. I hope these codes are helpful to you!
I would like to thank STIDGCN for the code instructions！

<p align="center">
  <img src="TEDDCF/figs/fig1.jpg" width="100%">
  <img src="TEDDCF/figs/fig2.jpg" width="100%">
</p>

## Requirements
The code is built based on Python 3.9.12, PyTorch 1.11.0, and NumPy 1.21.2.

## Datasets #生成数据函数是STSGCN的方法
We provide preprocessed datasets that you can access [here](https://drive.google.com/drive/folders/1-5hKD4hKd0eRdagm4MBW1g5kjH5qgmHR?usp=sharing). If you need the original datasets, please refer to [STSGCN](https://github.com/Davidham3/STSGCN) (including PEMS03, PEMS04, PEMS07, and PEMS08) and [ESG](https://github.com/LiuZH-19/ESG) (including NYCBike and NYCTaxi).

## Train Commands
It's easy to run! Here are some examples, and you can customize the model settings in run.ipynb.
### PEMS08
```
%run train.py --data PEMS08 --epochs 200 --save logs/name
```
### NYCBike Drop-off
```
%run train.py --data bike_drop --epochs 200 --save logs/name
```

## Acknowledgments
Our model is built based on model of [STIDGCN](https://github.com/LiuAoyu1998/STIDGCN) and [STwave](https://github.com/lmissher/stwave).
