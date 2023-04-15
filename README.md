# Towards Hard-pose Virtual Try-on via 3D-aware Global Correspondence Learning
Official implementation for NeurIPS 2022 paper "Towards Hard-pose Virtual Try-on via 3D-aware Global Correspondence Learning"

## Requirements

* python 3.8.12
* pytorch 1.10.2
* cudatoolkit 11
* opencv-python, scikit-image


## Data Preparation
We train our model on famous [Deepfashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) Dataset. The keypoints and human parsings are obtained using [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy). We follow the train test split of [GFLA](https://github.com/RenYurui/Global-Flow-Local-Attention) and [Pose with Style](https://github.com/BadourAlBahar/pose-with-style) as mentioned in the paper. Please download the dataset and organize the data structure as:

```
| Deepfashion_512_320
|   | image
|       | e.g. image1.jpg
|       | ...
|   | keypoints
|       | e.g. image1_keypoints.json
|       | ...
|   | parsing
|       | e.g. image1.png
|       | ...
|   | train_512.csv
|   | test_512.csv
```


## Inference
Download the [pre-trained models](https://drive.google.com/drive/folders/1CRDmpF04khb1icjse6Q-pdBx5zO9R9B3?usp=share_link) and then run the following command to get inference results:
```
CUDA_VISIBLE_DEVICES=0 python test.py --name test --phase test --dataset_mode smpl512psw --gpu_ids 0, \ --batchSize 1 --model test --netG Stylegan2 --dataroot YOUR_DATA_PATH/Deepfashion_512_320 --netCorr GlobalHD
```
we also provide a simple bash script to test the pre-trained model:
```
bash test.sh
```

## Training
To train our model from scratch:
1. Download and prepare the Deepfashion dataset.
2. Run the following command to train the warping model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage.py --name train_warp --phase train --dataset_mode smpl512psw --gpu_ids 0,1,2,3 --batchSize 8 --model GlobalCorrespondence --netG Stylegan2 --dataroot YOUR_DATA_PATH/Deepfashion_512_320 --netCorr GlobalHD 
```
3. Run the following command to train the fusion model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_tryon.py --name train_tryon --phase train --dataset_mode smpl512psw --gpu_ids 0,1,2,3 --batchSize 8 --model StyleGAN2Tryon --netG Stylegan2 --dataroot YOUR_DATA_PATH/Deepfashion_512_320 --netCorr GlobalHD 
```
We also provide the training script, run:
```
run train_stage.sh
run train_tryon.sh
```


## Acknowledgement
This code borrow heavily from [CocosNet-v2](https://github.com/microsoft/CoCosNet-v2) and [Pose with style](https://github.com/BadourAlBahar/pose-with-style), we really appreciate their work and would like to thank them for sharing the code.
