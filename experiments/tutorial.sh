#/bin/bash

DIR=ASPEC100k
src=en
tgt=ja
log_folder=tutorial/base
GPU=0
BATCH_SIZE=50
train=train
valid=dev
test=test

#言語順方向
log_folder1=$log_folder/${src}-${tgt}
mkdir -p ../OUT/$log_folder1
pushd ../
python main.py \
    -cuda_n $GPU \
    -save $log_folder1 \
    -batch_size $BATCH_SIZE \
    -train_src $DIR/$SRC_TOKEN/$train.$src \
    -train_trg $DIR/$TGT_TOKEN/$train.$tgt \
    -valid_src $DIR/$SRC_TOKEN/$valid.$src \
    -valid_trg $DIR/$TGT_TOKEN/$valid.$tgt \
    -test_src $DIR/$SRC_TOKEN/$test.$src \
    -test_trg $DIR/$TGT_TOKEN/$test.$tgt  > OUT/$log_folder1/log.out
popd
