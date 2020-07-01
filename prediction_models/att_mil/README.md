### Start docker image
 nvidia-docker run -it --rm --ipc=host -e NVIDIA_VISIBLE_DEVICES=7 --shm-size 4G -v /raid/jiayunli/projects/PANDA_Challenge/:/PANDA_Challenge -v /raid/jiayunli/data/PANDA_challenge/:/data panda-jiayun
 
 python -m prediction_models.att_mil.main_trainval --cuda --num_workers 4 --data_dir /data/ --dataset br_256_256 --cache_dir ./cache/br_256_256/  --im_size 256 --input_size 256  --top_n 32 --arch efficientnet-b0 --lr 0.0003 --schedule_type cosine --loss_type bce --batch_size 8 --exp effieicntnet_3e-4_bce_256
 
 python -m prediction_models.att_mil.main_trainval --cuda --num_workers 4 --data_dir /data/ --dataset br_256_2x  --cache_dir ./cache/br_256_2x/  --im_size 256 --input_size 256  --top_n 32 --arch efficientnet-b0 --lr 0.0003 --schedule_type cosine --loss_type bce
 
 #### Training with selected tiles
python -m prediction_models.att_mil.main_trainval --cuda --num_workers 4 --data_dir /slides_data/ --dataset selected_10x  --cache_dir ./cache/selected_10x/  --input_size 256  --top_n 36 --lr 0.0003 --schedule_type cosine --loss_type bce --batch_size 6 --top_n 36

python -m prediction_models.att_mil.test_model --data_dir /slides_data/train_images/ --model_dir /data/storage_slides/PANDA_challenge/trimmed_weights/resenet50_bce_newsplit_has02   --att_dir ./info/att_selected/ --cuda

python -m prediction_models.att_mil.datasets.get_selected_locs --data_dir /slides_data/train_images/ --info_dir ./info/new_split/ --select_model resenet50_bce_newsplit_has02 --select_method orig --high_res_fov 64 --select_per 0.25  --att_dir ./info/att_selected/