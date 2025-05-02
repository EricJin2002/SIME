# to train on single GPU
cd ../src && torchrun --standalone --nproc_per_node=gpu \
 train.py --config /path/to/config.json
 
# to train on multiple GPUs
cd ../src && CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 \
 --rdzv_backend c10d --rdzv_endpoint localhost:0 --nproc_per_node=gpu \
 train.py --config /path/to/config.json