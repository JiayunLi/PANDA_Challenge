### Start docker image
 nvidia-docker run -it --rm --ipc=host -e NVIDIA_VISIBLE_DEVICES=7 --shm-size 4G -v /raid/jiayunli/projects/PANDA_Challenge/:/PANDA_Challenge -v /raid/jiayunli/data/PANDA_challenge/:/data panda-jiayun
 
 python -m prediction_models.att_mil.main_trainval --cuda --num_workers 4 --data_dir /data/ --dataset br_256_256 --cache_dir ./cache/br_256_256/  --im_size 256 --input_size 256  --top_n 32 --arch efficientnet-b0 --lr 0.0003 --schedule_type cosine --loss_type bce --batch_size 8 --exp effieicntnet_3e-4_bce_256
 
 python -m prediction_models.att_mil.main_trainval --cuda --num_workers 4 --data_dir /data/ --dataset br_256_2x  --cache_dir ./cache/br_256_2x/  --im_size 256 --input_size 256  --top_n 32 --arch efficientnet-b0 --lr 0.0003 --schedule_type cosine --loss_type bce