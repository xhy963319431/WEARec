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

### How to test pretrained BSARec
- Note that pretrained model (.pt file) must be in `src/output`
- `load_model`: pretrained model name without .pt
```
python main.py  --data_name [DATASET] \
                --lr [LEARNING_RATE] \
                --alpha [ALPHA] \ 
                --num_heads [N_HEADS] \
                --train_name [LOG_NAME]
                --do_eval
```
- Example for LastFM
```
python main.py  --data_name LastFM \
                --lr 0.001 \
                --alpha 0.3 \
                --num_heads 2 \
                --train_name WEARec_K_50_LastFM_0.5_0.001_0.3_2
                --do_eval
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

## Citation
If you find our work useful, please consider citing our paper:
```
@misc{xu2025waveletenhancedadaptivefrequency,
      title={Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation}, 
      author={Huayang Xu and Huanhuan Yuan and Guanfeng Liu and Junhua Fang and Lei Zhao and Pengpeng Zhao},
      year={2025},
      eprint={2511.07028},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2511.07028}, 
}
```

## Contact
If you have any inquiries regarding our paper or codes, feel free to reach out via email at xuhuayang2001@163.com.

## Acknowledgement
This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).
