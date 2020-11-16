python generate_cmlm.py ../data/data-bin/ --path ../models/iwslt-word/checkpoint_best.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 1 --decoding-strategy mask_predict
mv translation.txt translation/translation1.txt
python generate_cmlm.py ../data/data-bin/ --path ../models/iwslt-word/checkpoint_best.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 3 --decoding-strategy mask_predict
mv translation.txt translation/translation3.txt
python generate_cmlm.py ../data/data-bin/ --path ../models/iwslt-word/checkpoint_best.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 5 --decoding-strategy mask_predict
mv translation.txt translation/translation5.txt
python generate_cmlm.py ../data/data-bin/ --path ../models/iwslt-word/checkpoint_best.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 7 --decoding-strategy mask_predict
mv translation.txt translation/translation7.txt
python generate_cmlm.py ../data/data-bin/ --path ../models/iwslt-word/checkpoint_best.pt  --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 9 --decoding-strategy mask_predict
mv translation.txt translation/translation9.txt
