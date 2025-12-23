# DeepTrace: GNN-Powered Detection and Intervention for Deepfake Propagation in Social Networks

## Overview
This project detects deepfake videos using spatio-temporal GNNs and simulates their propagation in social networks for intervention analysis. Datasets: FF++ c23, Celeb-DF v2, DFDC subset.

## Setup
1. Download datasets: bash scripts/download_datasets.sh
2. Install deps: pip install -r requirements.txt
3. Preprocess: python src/data_loader.py --dataset all  # Builds graphs in data/processed/
4. Train detection: python src/train_detection.py --config config.yaml
5. Evaluate: python src/evaluate_detection.py --model results/models/detection_gnn.pt
6. Propagation sim: python src/propagation_sim.py

## Datasets
- FF++ c23: https://www.kaggle.com/datasets/xdxd003/ff-c23
- Celeb-DF v2: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
- Deepforenscics subset: https://www.kaggle.com/competitions/deepfake-detection-challenge/data

## License
MIT
