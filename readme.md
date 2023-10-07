
# Create dataset
```
cd data/DfT4D
python preprocess.py --datapath ./ --human bear --pose pose
```

# Train

## Base model (initialization)

```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/RGBD/ad_RGBD_grad01_lr0001.yaml --mode train --rep sdf --continue_from 1499

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --config ./config/RGBD/ad_RGBD_grad01_lr0001.yaml --mode train --rep sdf --batch_size 3 --continue_from 1499

```
CUDA_VISIBLE_DEVICES=1,2,3 bash scripts/dist_train.sh 3 --config ./config/RGBD/kfusion_RGBD_grad01_lr0001.yaml --mode train --rep sdf --batch_size 10
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf --continue_from 4999
```

## Enforce ARAP
Change params: `use_sdf_asap_epoch` in the `yaml` file. Tune `lr` and `batch_size`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf --batch_size 2 --continue_from 4999
```

# Train 2

## Step 1: Base model (initialization)
### merge all the training meshes and fit them --> form an initialization

## Step 2: fit each mesh

## Step 3: ARAP
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf --continue_from 0 --train_from_merge
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf --batch_size 4 --continue_from 0 --train_from_merge
```

# Interpolation
```

CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/RGBD/ad_RGBD_grad01_lr0001.yaml --mode interp --rep sdf --continue_from 4499 --split train
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/RGBD/kfusion_RGBD_grad01_lr0001.yaml --mode interp --rep sdf --continue_from 4499 --split train

If `interp_src_fid` and `interp_tgt_fid` are not specified, then by default we interpolate the longest sequence.

```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode interp --rep sdf --continue_from 4499 --split train --interp_src_fid 0 --interp_tgt_fid 1
```

# Evaluate Energy
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode evaluate --rep sdf --continue_from 9999 --split train
```

evaluate partial trained model with full trained model



random warp, 比如t = t1, t = t2, 知道他们的translation matrix，我在t1随机sample一些点，得到SDF，然后translaation回t2，再得到一批SDF，尽可能小。