train_gpus_id=0,1,2,3
eva_gpus_id=0
train_list_path='./Datasets/image_test.pt'
evalution_format='training'
echo "test gpus id: "${train_gpus_id}
echo "the list path is: "${train_list_path}
echo "start to predict disparity map"

CUDA_VISIBLE_DEVICES=${train_gpus_id} python -u Source/main.py \
                        --mode train \
                        --batchSize 4 \
                        --gpu 2 \
                        --trainListPath ${train_list_path} \
                        --auto_save_num 1 \
                        --dataloaderNum 8 \
                        --imgNum 8145177 \
                        --maxEpochs 50 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --dist True \
                        --modelName TM_Model_A \
                        --dataset tmdataset \
                        --log ./log/ \
                        --outputDir ./result/\
                        --resultImgDir ./ResultImg/ \
                        --modelDir ./Checkpoint/ > ./start_train_dataset_model.log 2>&1 &
echo "Finish!"
tail -f ./start_train_dataset_model.log