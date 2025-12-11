#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

# Remove directories if they exist
Remove-Item -Recurse -Force "filtered_data" -ErrorAction SilentlyContinue

Write-Host "Begin Preprocessing"
python preprocess.py

Write-Host "Train Empty Mask Classifier"
python train_empty_detector.py

Write-Host "Filter Empty Mask Images"
python filter_empty_images.py

Write-Host "Augment Data Sequences"
python augment2d_data_per_sequence.py

Write-Host "Preprocessing Finished"

# Cleanup
Remove-Item -Recurse -Force "processed_data" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "__pycache__" -ErrorAction SilentlyContinue
