#!/bin/sh
correct="../Corpus/ASPEC-JE/corpus100k/test.ja"
output="RESULT/nonsubword/output.txt"
~/tool/moses/mosesdecoder/scripts/generic/multi-bleu.perl $correct < $output
