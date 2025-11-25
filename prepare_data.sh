#!/bin/bash

set -e
rm -rf filtered_data
echo "Begin Preprocessing"
python preprocess.py

echo "Train Empty Mask Classifier"
python train_empty_detector.py

echo "Filter Empty Mask Images"
python filter_empty_images.py

echo "Preprocessing Finished"
rm -rf processed_data
rm -rf __pycache__