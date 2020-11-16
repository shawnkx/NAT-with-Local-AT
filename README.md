# Incorporating a Local Translation Mechanism into Non-autoregressive Translation
Code for the paper
[Incorporating a Local Translation Mechanism into Non-autoregressive Translation](https://arxiv.org/pdf/2011.06132.pdf) Xiang Kong*, Zhisong Zhang*, Eduard Hovy

## Usage
### Training a CMLM with LAT
```bash
data_bin=PATH_To_DATA
model_dir=PATH_To_STORE_MODEL
python train.py ${data_bin} --arch bert_transformer_seq2seq_rnn  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self_rnn_context --max-tokens 4000 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${model_dir} --len-context 2 --context-dir right --keep-last-epochs 10  --unmask-loss-ratio 0.1 --share-all-embeddings  --update-freq 16 --save-interval-updates 1000 --no-epoch-checkpoints --keep-interval-updates 20 --ddp-backend=no_c10d
```
### Test the model
```bash
python generate_cmlm.py ${data_bin} --path ${model_dir}/checkpoint_best.pt  --task translation_self_rnn_context  --remove-bpe --max-sentences 20 --decoding-iterations 1 --decoding-strategy mask_predict_rnn
```
## Reference
If you find our data or code useful, please consider citing our paper:
```
@inproceedings{kong2020incorporating,
  title={Incorporating a Local Translation Mechanism into Non-autoregressive Translation},
  author={Kong, Xiang and Zhang, Zhisong and Hovy, Eduard},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={1067--1073},
  year={2020}
}
```

## Acknowledgement
* The code is adapted from MaskPredict (https://github.com/facebookresearch/Mask-Predict). Thanks!


