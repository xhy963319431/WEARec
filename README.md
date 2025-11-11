# WEARec
This is the official source code for our AAAI 2026 Paper ["Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation"](https://arxiv.org/abs/2511.07028)

## Dataset
In our experiments, we utilize four datasets, all stored in the `src/data` folder. Given the limited space for uploaded files, we have only uploaded one dataset, LastFM, as an example.
- The `src/data/*_same_target.npy` files are utilized for training DuoRec, SLIME4Rec and FEARec, both of which incorporate contrastive learning.

## Quick Start
### Environment Setting
```
conda env create -f wearec_env.yaml
conda activate wearec
```

### How to train WEARec
- Note that pretrained model (.pt) and train log file (.log) will saved in `src/output`
- `train_name`: name for log file and checkpoint file
```
python main.py  --data_name [DATASET] \
                --lr [LEARNING_RATE] \
                --alpha [ALPHA] \ 
                --num_heads [N_HEADS] \
                --train_name [LOG_NAME]
```
- Example for LastFM
```
python main.py  --data_name LastFM \
                --lr 0.001 \
                --alpha 0.3 \
                --num_heads 2 \
                --train_name WEARec_LastFM
```


### How to train the baselines
- You can easily train the baseline models used in WEARec by changing the `model_type` argument.
    - `model_type`: Caser, GRU4Rec, SASRec, BERT4Rec, FMLPRec, DuoRec, SLIME4Rec, FEARec, BSARec
- For the hyperparameters for the baselines, check the `parse_args()` function in `src/utils.py`.
```
python main.py  --model_type SASRec \
                --data_name LastFM \
                --num_attention_heads 1 \
                --train_name SASRec_LastFM
```


