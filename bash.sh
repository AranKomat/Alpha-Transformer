#!/bin/bash
for number in {0..5}
do
CUDA_VISIBLE_DEVICES=$[$number % 3] python3 self --process_num=$number
done
CUDA_VISIBLE_DEVICES=3 python3 opt
exit 0

