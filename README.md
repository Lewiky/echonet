# Environment Sound Classification: Replication of Work
## COMSM0018: Applied Deep Learning 

This project contains the PyTorch code for training a CNN to classify the UrbanSound8K dataset.
The default data directory is `./data`, where the `UrbanSound8K_test.pkl` and `UrbanSound8K_train.pkl` files are expected.
This can be modified with the `--dataset-root` flag.

### BC4
BlueCrystal 4 SLURM scripts are included with this project for training all models, including an additional script `train-lmc-aug.sh` which demonstrates the data augmentation extension.

### Usage
```
❯❯❯ python src/main.py --help
usage: main.py [-h] [--dataset-root DATASET_ROOT] [--log-dir LOG_DIR]
               [--qual-results QUAL_RESULTS] [--learning-rate LEARNING_RATE]
               [--batch-size BATCH_SIZE] [--epochs EPOCHS]
               [--val-frequency VAL_FREQUENCY] [--log-frequency LOG_FREQUENCY]
               [--print-frequency PRINT_FREQUENCY] [-j WORKER_COUNT]
               [--mode [{LMC,MC,MLMC,TSCNN}]] [--dropout [DROPOUT]]
               [--weight-decay [WEIGHT_DECAY]]
               [--augmentation-length [AUGMENTATION_LENGTH]]

Train a network for performing intelligent sound recognition using the
UrbanSound8K dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset-root DATASET_ROOT
  --log-dir LOG_DIR
  --qual-results QUAL_RESULTS
                        File path to store qualitative results output
                        (default: None)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --batch-size BATCH_SIZE
                        Number of images within each mini-batch (default: 32)
  --epochs EPOCHS       Number of epochs (passes through the entire dataset)
                        to train for (default: 50)
  --val-frequency VAL_FREQUENCY
                        How frequently to test the model on the validation set
                        in number of epochs (default: 2)
  --log-frequency LOG_FREQUENCY
                        How frequently to save logs to tensorboard in number
                        of steps (default: 10)
  --print-frequency PRINT_FREQUENCY
                        How frequently to print progress to the command line
                        in number of steps (default: 10)
  -j WORKER_COUNT, --worker-count WORKER_COUNT
                        Number of worker processes used to load data.
                        (default: 4)
  --mode [{LMC,MC,MLMC,TSCNN}]
  --dropout [DROPOUT]   Dropout probability propagated to all networks
                        (default: 0.5)
  --weight-decay [WEIGHT_DECAY]
                        Weight decay to apply to Adam optimiser (default:
                        0.0001)
  --augmentation-length [AUGMENTATION_LENGTH]
                        Total iterations through dataset - >1 will include
                        augmentation (default: 1)
```
### Install Dependencies

```
pip3 install -r requirements.txt
```
