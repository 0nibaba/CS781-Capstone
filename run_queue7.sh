python -u train_pcr.py --run_name pcr_lorma --finetune lorma \
  --checkpoint ../checkpoints/arch2_prefix_3B/ckpt_best.pt \
  --max_lr 3e-4 --epochs 6 > ../logs/pcr_lorma.log 2>&1

python -u train_pcr.py --run_name pcr_topk --finetune top_k --top_k_layers 4 \
  --checkpoint ../checkpoints/arch3_efficient/ckpt_best.pt \
  --max_lr 1e-4 --epochs 6 > ../logs/pcr_topk.log 2>&1
