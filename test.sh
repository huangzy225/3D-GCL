CUDA_VISIBLE_DEVICES=0 python test.py --name test --phase test --dataset_mode smpl512psw --gpu_ids 0, \ --batchSize 1 --model test --netG Stylegan2 --dataroot YOUR_DATA_PATH/Deepfashion_512_320 --netCorr GlobalHD