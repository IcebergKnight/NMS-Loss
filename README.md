# NMS-Loss
[NMS-Loss: Learning with Non-Maximum Suppression for Crowded Pedestrian Detection
](https://arxiv.org/abs/2106.02426)

## Introduction

NMS-Loss test on Citypersons和Caltech：

|	dataset		| Config   	| MR  		|
|------------		|:--------:	|:--------:	|
|Citypersons	| cityperons.py | 10.08% |
|Caltech(Ori)	| caltech.py | 5.92% |

## Installation
Prerequisites:

* Linux (Windows is not officially supported)
* Python 3.5+
* PyTorch 1.1
* CUDA 9.0 or higher
* NCCL 2
* GCC 4.9 or higher
* [mmcv==0.2.16](https://github.com/open-mmlab/mmcv)

a. Create a conda virtual environment and activate it.

```
conda create -n nms-loss python=3.7 -y
conda activate nms-loss
```
b. Install PyTorch, torchvision and mmcv

```
conda install pytorch=1.1.0 torchvision
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install mmcv==0.2.16
```
c. Clone

```
git clone http://git.code.oa.com/zekunluo/nms-loss.git
cd nms-loss
```
d. Check GCC, if GCC < 4.9:

```
conda install -c psic4 gcc5
```

e. install

```source-shell
./compile.sh
pip install -v -e .  # or "python setup.py develop"
```
## Test

Dowdload weights from https://drive.google.com/drive/folders/1MwdnknqX6I3lNIbMVJQOyVxGK1lw-dEX?usp=sharing.

Citypersons:
```
./tools/dist_test.sh configs/cityperons.py work_dirs/citypersons.pth 8 --out results/citypersons.pkl --eval bbox
python3 tools/eval_script/eval_demo.py
```

Caltech:
```
./tools/dist_test.sh configs/caltech.py work_dirs/caltech.pth 8 --out results/caltech.pkl --eval bbox
python3 tools/caltech_pkl2txt.py
```
