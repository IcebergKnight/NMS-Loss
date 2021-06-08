conda create -n nms-loss python=3.7 -y
conda activate nms-loss

conda install pytorch==1.1.0 torchvision==0.3.0
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install mmcv==0.2.16 xgboost

cd nms-loss
./compile.sh
python3 setup.py develop

./tools/dist_test.sh configs/cityperons.py work_dirs/citypersons.pth 8 --out results/citypersons.pkl --eval bbox
python3 tools/eval_script/eval_demo.py

./tools/dist_test.sh configs/caltech.py work_dirs/caltech.pth 8 --out results/caltech.pkl --eval bbox
python3 tools/caltech_pkl2txt.py