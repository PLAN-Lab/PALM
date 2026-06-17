# Running
## Notice

For convenience, some checkpoints, such as the MAE-pretrained ViT-B model, are provided for manual download. Users must update the following paths accordingly. Relevant checkpoints can be acquired from the [website](https://drive.google.com/drive/folders/1zwqGvKKtjyuWdDaNSLVGJprJMPoSqAPk?usp=drive_link).
* :exclamation: **pretrain.sh, finetune.sh, scratch, eval.sh:**
Please update the following:
    * **save_checkpoint_path** to the parent directory where your experiment checkpoints are saved.  Recommend to create a ```checkpoints``` folder in the project root directory.
    * **finetune_from_pretrained_ckpt** to the location of your pre-trained checkpoint.
    * **resume_from_checkpoint** to the location of your fine-tuned checkpoint.
    * **vit_checkpoint_path** to the location of your ViT checkpoint (downloaded from the [website](https://drive.google.com/file/d/1bSsvRI4mDM3Gg51C6xO0l9CbojYw3OEt/view?usp=sharing)). Recommend to be stored in ```checkpoints/vit_mae/mae_pretrain_vit_base.pth```.
    * **libero_path** to the location of LIBERO dir.

## PALM
### Convert Data
```bash
python utils/convert_libero_per_step.py
```

### Pre-train
```bash
# Pre-train PALM on LIBERO-90 dataset
bash scripts/LIBERO/PALM/pretrain.sh
```

### Fine-tune
```bash
# Fine-tune PALM on LIBERO-Goal, LIBERO-Object, LIBERO-Spatial, LIBERO-10 datasets
bash scripts/LIBERO/PALM/finetune_goal.sh
bash scripts/LIBERO/PALM/finetune_object.sh
bash scripts/LIBERO/PALM/finetune_spatial.sh
bash scripts/LIBERO/PALM/finetune_long.sh
```

### Train from Scratch
```bash
# Train PALM on LIBERO-10 dataset from scratch
bash scripts/LIBERO/PALM/scratch.sh
```

### Eval
```bash
# Evaluate PALM on LIBERO-Goal, LIBERO-Object, LIBERO-Spatial, LIBERO-10 benchmarks
bash scripts/LIBERO/PALM/eval_goal.sh
bash scripts/LIBERO/PALM/eval_object.sh
bash scripts/LIBERO/PALM/eval_spatial.sh
bash scripts/LIBERO/PALM/eval_long.sh
```
