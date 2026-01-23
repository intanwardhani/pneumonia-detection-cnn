# Distinguishing Pneumonic From Healthy Lungs

In this project, we built a deep-learning system to assist medical professional in distinguishing healthy lungs from lungs infected with pneumonia. 

# Authors
[Intan K. Wardhani](https://github.com/intanwardhani) and [Sandrine Herbelet](https://github.com/Sandrine111222). 

# Workflow


# Usage

# Project Structure
```
pneumonia-detection-cnn/
├── README.md
├── requirements.txt
├── environment.yml
├── config.py
├── main.py
├── train.py
├── evaluate.py
├── predict.py
├── data/
│   ├── raw/
│   │   ├── chest_xray/
│   │   │   ├── train/
│   │   │   │   ├── NORMAL/
│   │   │   │   │   ├── img1.jpeg
│   │   │   │   │   └── img2.jpeg
│   │   │   │   └── PNEUMONIA/
│   │   │   │       ├── img3.jpeg
│   │   │   │       └── img4.jpeg
│   │   │   ├── val/
│   │   │   │   ├── NORMAL/
│   │   │   │   └── PNEUMONIA/
│   │   │   └── test/
│   │   │       ├── NORMAL/
│   │   │       └── PNEUMONIA/
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── external/
│       ├── additional_xrays/
│       └── metadata/
│           └── patient_info.csv
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transfer_learning.py
│   │   ├── custom_cnn.py
│   │   └── model_factory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── visualizations.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── helpers.py
│       └── config_parser.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_experiments.ipynb
│   ├── visualization_analysis.ipynb
│   └── hyperparameter_tuning.ipynb
├── outputs/
│   ├── models/
│   │   ├── resnet18_pneumonia.pth
│   │   ├── resnet50_pneumonia.pth
│   │   └── densenet121_pneumonia.pth
│   ├── logs/
│   │   ├── training.log
│   │   └── validation.log
│   ├── results/
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_history.png
│   │   └── classification_report.txt
│   ├── predictions/
│   │   ├── test_predictions.csv
│   │   └── sample_predictions.json
│   └── checkpoints/
│       ├── epoch_10.pth
│       └── epoch_20.pth
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_evaluation.py
├── docs/
│   ├── project_overview.md
│   ├── data_preprocessing.md
│   ├── model_architecture.md
│   ├── training_configuration.md
│   ├── evaluation_metrics.md
│   └── deployment_guide.md
├── scripts/
│   ├── download_data.sh
│   ├── preprocess_data.py
│   ├── train_model.sh
│   ├── evaluate_model.sh
│   └── clean_data.sh
├── config/
│   ├── default.yaml
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   └── densenet121.yaml
├── .gitignore
├── .gitattributes
├── Dockerfile
└── docker-compose.yml
```


