CUDA_VISIBLE_DEVICES=0,1,2,3 python train_stage.py --name train_warp --phase train --dataset_mode smpl512psw --gpu_ids 0,1,2,3 --batchSize 8 --model GlobalCorrespondence --netG Stylegan2 --dataroot YOUR_DATA_PATH/Deepfashion_512_320 --netCorr GlobalHD 