
# 检查错误使用：
# RuntimeError: Expected to have finished reduction in the prior iteration
# TORCH_DISTRIBUTED_DEBUG=DETAIL python train_pl.py






export NCCL_P2P_DISABLE=1 

# 正常训练使用
nohup python ./main/train.py --configs ./configs/config_instance.py >$1.log 2>&1 &


# tail -10f $1.log
