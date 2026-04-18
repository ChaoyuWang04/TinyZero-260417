#!/bin/bash
#
# ============================================================================
# TinyZero GRPO Training Script - Qwen2.5-3B on Countdown Task
# ============================================================================
set -x

# ============================================================================
# 【SECTION A】硬件与路径配置
# ============================================================================
# 换 pod、换模型、换数据集时必改。启动前务必验证路径存在。
# ============================================================================

# ▼ 数据和模型路径
DATA_DIR=/workspace/TinyZero-260417/data
MODEL_PATH=/workspace/TinyZero-260417/models/Qwen2.5-3B

# ▼ GPU 配置
# 改变时同步改 TP_SIZE（见下）
N_GPUS=2                    # 实际可用的 GPU 数量，nvidia-smi 能看到几张写几张
NNODES=1                    # 节点数，单机就是 1

# ▼ vLLM Tensor Parallelism
# 规则：TP_SIZE 必须能整除 N_GPUS
#   N_GPUS=2: TP_SIZE=2（单 rollout 实例）
#   N_GPUS=4: TP_SIZE=2（推荐，2 个 rollout 实例并行）
#   N_GPUS=4: TP_SIZE=4（单实例，大模型用）
#   N_GPUS=8: TP_SIZE=2 或 4
TP_SIZE=2

# ============================================================================
# 【SECTION B】训练规模与 GRPO 核心参数
# ============================================================================
# 这些是影响"能不能学会"的关键参数。第一次跑保守，看曲线后调整。
# ============================================================================

# ▼ GRPO 组大小（重要性：★★★★★）
# 每个 prompt 生成几条 trajectory，组内算 advantage
#   n=4:  TinyZero 默认，显存紧张时用
#   n=8:  推荐值，平衡效果和速度
#   n=16: DeepSeek-R1 原论文值，最稳但贵
# 显存和时间随 n 线性增长
ROLLOUT_N=4

# ▼ 每 step 处理多少个 prompt（重要性：★★★★★）
# 全局 batch size，决定梯度稳定性
# 显存压力 ∝ batch_size × rollout_n × max_response_length
# 一般范围 128-512，Countdown 任务 256 足够
TRAIN_BATCH_SIZE=128

# ▼ Response 长度上限（重要性：★★★★）
# 模型生成的 <think>...</think><answer>...</answer> 总长度
# 对 reasoning 任务，太短会截断思考过程；太长浪费显存
#   Countdown:  1024 足够
#   复杂数学:    2048-4096
#   Coding:    4096+
MAX_RESPONSE_LEN=512

# ▼ Prompt 长度上限
# Countdown 的 prompt 很短，256 够用
MAX_PROMPT_LEN=256

# ▼ 训练总量控制（重要性：★★★★）
# TinyZero 原作者只跑了 200-500 步就看到 Aha moment
# 推荐：先用 total_training_steps=500 观察曲线，看到 reward 起飞就 Ctrl+C
# 如果没有 total_training_steps 参数（看 log 确认），用 total_epochs=1（≈1280 steps）
TOTAL_EPOCHS=15            # 真正跑完会到 19200 步，但实际 500 步就应该能看到效果

# ============================================================================
# 【SECTION C】显存与性能调优
# ============================================================================
# OOM 或速度问题时调这里。原则：先调 token_len，后开 offload。
# ============================================================================

# ▼ vLLM 显存占比（重要性：★★★）
# vLLM 占用 GPU 显存的比例（剩余留给 training）
#   0.3: 显存紧张（2 卡 3B 必须）
#   0.5: 标准
#   0.7: 显存充裕（4 卡+）
# 越大 KV cache 越大，rollout 越快
ROLLOUT_GPU_UTIL=0.3

# ▼ 每 GPU 动态 batch token 上限（重要性：★★★）
# use_dynamic_bsz=True 时生效，决定每卡一次处理多少 token
# OOM 时先降这个（比降 batch_size 影响小）
#   6000:  2 卡 3B + offload，保守
#   8000:  2 卡 3B + offload，标准
#   12000: 4 卡 3B 或 2 卡 + 激进优化
#   16000+: 大显存或小模型
PPO_MAX_TOKEN_LEN=4096

# ▼ 是否 offload Actor 到 CPU（重要性：★★★）
# True: 省 ~18 GB 显存（param + optimizer），慢 20-40%
# False: 快，但显存吃紧
# 2 卡 3B 必须开，4 卡可以试关
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True

# ▼ 梯度检查点（重要性：★★）
# True: 省 ~30% activation 显存，慢 20%
# False: 快，但显存吃紧
# 一般都开
GRADIENT_CHECKPOINTING=True

# ▼ PPO update 的 mini batch
# 全局 batch 在 PPO update 时切成多少个 mini batch
# train_batch_size × rollout.n / ppo_mini_batch_size = PPO update 的 micro step 数
PPO_MINI_BATCH=64
PPO_MICRO_BATCH=8

# ============================================================================
# 【SECTION D】学习率与 KL 超参
# ============================================================================
# 训练稳定后很少改，除非出现"学不会"或"崩盘"
# ============================================================================

# ▼ Actor 学习率（重要性：★★★★★，但不要乱改）
# RL 的 lr 比 SFT 小 100-1000 倍
# 1e-6 是 TinyZero 作者和 DeepSeek-R1 验证过的
# 改动范围：5e-7 到 5e-6
ACTOR_LR=1e-6

# ▼ KL 惩罚系数（重要性：★★★★★，但不要乱改）
# 约束 policy 不要偏离 reference model 太远
# 0.001 是标准值
# 太大（>0.01）：模型不敢探索，学不会
# 太小（<0.0001）：模型可能崩溃，输出怪异
KL_LOSS_COEF=0.001
KL_CTRL_COEF=0.001

# ============================================================================
# 【SECTION E】日志与持久化
# ============================================================================
# 换一个实验就改 EXPERIMENT_NAME，方便 WandB 对比
# ============================================================================

PROJECT_NAME=TinyZero
# 建议命名规则: <任务>-<模型>-<方法>-<GPU配置>-<版本>
EXPERIMENT_NAME=countdown-qwen2.5-3b-grpo-${N_GPUS}gpu-v3

# Checkpoint 频率（步数），太密会撑爆磁盘，太稀可能丢失进度
SAVE_FREQ=100

# 验证频率（步数），太密拖慢训练
TEST_FREQ=100

# ============================================================================
# 环境变量和校验
# ============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS  # verl + vllm 组合必须

# 启动前校验（路径存在性）
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "❌ ERROR: train.parquet 不存在于 ${DATA_DIR}"
    exit 1
fi
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "❌ ERROR: 模型不存在于 ${MODEL_PATH}"
    exit 1
fi

# 打印当前配置
echo "================================================================"
echo "Training config summary:"
echo "  GPUs:              ${N_GPUS} (TP=${TP_SIZE})"
echo "  Model:             ${MODEL_PATH}"
echo "  Data:              ${DATA_DIR}"
echo "  Batch size:        ${TRAIN_BATCH_SIZE}"
echo "  Rollout n:         ${ROLLOUT_N}"
echo "  Max resp length:   ${MAX_RESPONSE_LEN}"
echo "  LR:                ${ACTOR_LR}"
echo "  KL coef:           ${KL_LOSS_COEF}"
echo "  Experiment:        ${EXPERIMENT_NAME}"
echo "================================================================"

# ============================================================================
# 启动训练
# ============================================================================

/workspace/TinyZero-260417/.venv/bin/python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=${KL_CTRL_COEF} \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=1024 \
  data.max_prompt_length=${MAX_PROMPT_LEN} \
  data.max_response_length=${MAX_RESPONSE_LEN} \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING} \
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
  actor_rollout_ref.actor.ppo_micro_batch_size=${PPO_MICRO_BATCH} \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  +trainer.val_before_train=False \
  trainer.logger=['console','wandb'] \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=${NNODES} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.total_epochs=${TOTAL_EPOCHS}