# pytorch-mri-segmentation-3D
3D model implementations of:  
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;DeepLab v3 - [paper](https://arxiv.org/abs/1706.05587)  
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;UNET - [paper](https://arxiv.org/abs/1606.06650)  
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;HRNet - [paper](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28)  
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;EXPNet - experiment models DefaultCNN, PrivCNN  

# Usage
### Data structure
```
main_folder_path
│
│───scans
│	└───scan1
│		└───pre
│			└───FLAIR.nii.gz
│			wmh.nii.gz
│──────scan2
│		└───pre
│			└───FLAIR.nii.gz
│			wmh.nii.gz
│──────scan3
│		└───pre
│			└───FLAIR.nii.gz
│			wmh.nii.gz
```

### Setup
```
main_folder_path='../../Data/MS2017test/'
cd utils/
```
Train landmarks (for histogram normalization)
```
python trainLandmarks.py --mainFolderPath=$main_folder_path --pcRange=1-99
```
Resize and normalize scans (histogram normalization & per-subject normalization)
```
python resizeScans.py --mainFolderPath="$main_folder_path" --postfix=_200x200x100orig --noGIFS --withNorm=1 --size=200x200x100
```
Generate train/val split
```
python -c 'import PP;  PP.generateTrainValFile(0.8, main_folder = "'$main_folder_path'", postfix="_200x200x100orig")'
```
### Training
Train architecture 0 (DeepLab v3). Model is saved in train_results/models/
```
python train.py --useGPU=1 --archId=0 --maxIter=100000 --lr=0.0001 --iterSize=25 --patchSize=60
```
Train EXPNet model (D1xD2xD3xD4_priv_aspp)
```
python train.py --experiment=1x1x1x1_1_1 --useGPU=1 --maxIter=100000 --lr=0.0001 --iterSize=25 --patchSize=60
```
### Evaluation
Evaluate model. Results saved in eval_results/
```
python eval.py --patchPredSize=60 --modelPath=$modelpath --singleEval
```
### Requirements
PyTorch v0.2.0, Python 2.7