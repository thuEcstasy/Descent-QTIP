export MASTER_ADDR=localhost       # 主节点地址
export MASTER_PORT=12355           # 主节点端口，可以选择其他端口
export WORLD_SIZE=8                # 总进程数（8个GPU）
export RANK=0                      # 当前进程的 rank
export PYTHONPATH=$PYTHONPATH:/home/zs453/EfficientML/qtip/qtip-kernels

# 启动 8 个进程，每个进程对应一个 GPU
torchrun --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         --nnodes=1 --node_rank=0 input_hessian_llama.py 

