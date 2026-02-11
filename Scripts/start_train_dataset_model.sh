train_gpus_id=0
eva_gpus_id=0
train_list_path='./Datasets/image_train.csv'
evalution_format='training'
echo "test gpus id: "${train_gpus_id}
echo "the list path is: "${train_list_path}
echo "start to predict disparity map"
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 16 \
                        --gpu 1 \
                        --trainListPath ${train_list_path} \
                        --auto_save_num 5 \
                        --dataloaderNum 16 \
                        --imgNum 111360 \
                        --maxEpochs 5 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist False \
                        --modelName CSTM_Net \
                        --dataset CSTM_dataloader \
                        --log ./result/Train/log/ \
                        --outputDir ./result/Train/Result/ \
                        --resultImgDir ./result/Train/ResultImg/ \
                        --modelDir ./result/Checkpoint/ > ./result/Train/start_train_dataset_model.log 2>&1 &
echo "Finish!"