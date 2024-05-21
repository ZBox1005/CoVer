EXP_NAME=$2
ID=$3
SCORE=$4
MODEL=$1

CKPT=ViT-B-16
DATA_ROOT=datasets

CUDA_VISIBLE_DEVICES='0' \
python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --model ${MODEL} --CLIP_ckpt ${CKPT} --score ${SCORE} --root-dir ${DATA_ROOT} 
