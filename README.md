# Pizza / not-pizza image classification

A binary image classifier (pizza vs. not pizza) built with TensorFlow/Keras. Experiments are tracked in [ClearML](https://clear.ml/).

The project compares several CNN architectures and training strategies:

- **Baselines trained from scratch**: ResNet50, ResNet101, VGG16, InceptionV3 and MobileNet, plus ResNet runs with different batch and image sizes
- **Transfer learning**: ImageNet-pretrained backbones, with the base either frozen or fine-tuned
- **Regularisation**: dropout, light data augmentation (flip, rotation, zoom) and input normalisation
- **Hyperparameter search**: random search with ClearML's `HyperParameterOptimizer` over network, image size, dropout, augmentation, batch size and epochs, maximising validation accuracy

Each experiment script saves a confusion matrix next to it in `src/`.

## Layout

```
src/
├── utils.py                    # model factory (get_model) and confusion-matrix plotting
├── training_task.py            # single training run as a ClearML task
├── hpo_task.py                 # parameterised training task used by the optimiser
├── hpo_launcher.py             # ClearML random-search HPO launcher
├── clearml_dataset_upload.py   # uploads the pizza/not-pizza dataset to ClearML
└── <experiment>/               # one folder per experiment: script + confusion matrix
```

## Running

```bash
pip install -r requirements.txt
clearml-init                         # connect to your ClearML server
python src/clearml_dataset_upload.py # once, after pointing it at your dataset
python -m src.training_task          # single run
python src/hpo_launcher.py           # hyperparameter search (needs a ClearML agent on the "default" queue)
```

The scripts load the dataset by ClearML dataset ID. Replace the IDs with your own after uploading.
