
# Create dataset
```
cd data/DfT4D
python preprocess.py --datapath ./ --human bear --pose pose
```

# Train

## Base model (initialization)
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf
```

## Enforce ARAP
Change params: `use_sdf_asap_epoch` in the `yaml` file. Tune `lr` and `batch_size`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode train --rep sdf --continue_from 4499 --batch_size 2
```

# Interpolation
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/DfT4D/bear/tbase/ad_bear_grad01_lr0001.yaml --mode interp --rep sdf --continue_from 4499 --split train
```

If `interp_src_fid` and `interp_tgt_fid` are not specified, then by default we interpolate the longest sequence.

