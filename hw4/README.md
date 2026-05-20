Overview
===


# Train
Firstly, preprocess the data and save the processed data in `data/processed` by running:

```bash
python preprocess.py --num-workers 32
```
Then, you can train the model by running:
```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=4 train.py --variants baseline

torchrun --nproc_per_node=4 train.py --variants baseline,no_residual,pre_norm,moe
```
After training, you can find the checkpoints and metrics in `checkpoints/`.
You can also visualize the training curves by running:
```bash
python plot_metrics.py --inputs checkpoints/baseline/metrics.jsonl,checkpoints/no_residual/metrics.jsonl,checkpoints/pre_norm/metrics.jsonl,checkpoints/moe/metrics.jsonl \
--labels Baseline,No_Residual_Connection,Pre_Norm,MoEs
```
# Inference

```bash
python inference.py --checkpoint checkpoints/baseline/ckpt_step_600.pt --config checkpoints/baseline/config.json --text "这是一段测试文本，用于检测模型的中文到英文的翻译能力。" --max-new-tokens 128
```
