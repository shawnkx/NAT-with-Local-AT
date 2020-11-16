# output_dir=$1
# model_dir=$2
# #python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq_mlp  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self --max-tokens 8192 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${model_dir}
# for ratio in 0.05 0.5
# do
# for len_ctx in 1 3
# #for lr in 0.01
# do
# python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq_rnn  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self_rnn_context --max-tokens 8192 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 30000 --seed 0 --save-dir ${model_dir}-macro-del-0.1-${len_ctx}-${ratio} --len-context ${len_ctx} --context-dir right --keep-last-epochs 10  --unmask-loss-ratio ${ratio}
# done
# done

python train.py ../data/wmt14/wmt14-ende-dis/data-bin/ --arch bert_transformer_seq2seq_rnn  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self_rnn_context --max-tokens 4000 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ../models/wmt14-ende-models/wmt14-ende-dis-128k --len-context 2 --context-dir right --keep-last-epochs 10  --unmask-loss-ratio 0.1 --share-all-embeddings  --update-freq 16 --save-interval-updates 1000 --no-epoch-checkpoints --keep-interval-updates 20 --ddp-backend=no_c10d
