#!/usr/bin/env bash
python im2rec.py  --list --recursive ./data/train_dataset/didianwei-meituan-train-v4 ./data/ori_train_dataset/train
python im2rec.py  --list --recursive ./data/train_dataset/didianwei-meituan-val-v4 ./data/ori_train_dataset/val
python im2rec.py  --resize 224 --num-thread 4 ./data/train_dataset/didianwei-meituan-train-v4 ./data/ori_train_dataset/train
python im2rec.py  --resize 224 --num-thread 4 ./data/train_dataset/didianwei-meituan-val-v4 ./data/ori_train_dataset/val
