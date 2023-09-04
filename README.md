# Project 4 - Sim-to-Real transfer of Reinforcement Learning policies in robotics

- Francesco Carlucci - @Francesco-Carlucci
- Matteo Colucci - @MattColu
- Pietro Noto - @PietroNoto

## Usage

Argument list is available via `python main.py -h`

To train and test regular agents just run `python main.py --args...`

To train and test pose estimation agents:

- Run `git clone https://github.com/open-mmlab/mmpose <PROJECT_ROOT>/vision/mmpose`
- Download the trained pose estimation checkpoint from [this Google Drive link](https://drive.google.com/file/d/1_D-03IVNWFkfMc5FLXlGEqpDNuctU0h6/view?usp=sharing)
- Pass `--mmpose-config <PROJECT_ROOT>/vision/hopper_config_extra.py` and `--mmpose-checkpoint path/to/downloaded/checkpoint/epoch_100.pth` as arguments to `main.py`

To train the pose estimation network itself:

- Follow MMPose's [installation procedures](https://mmpose.readthedocs.io/en/latest/installation.html)
- Download the Hopper Dataset from [this Google Drive link](https://drive.google.com/drive/folders/1ejTVU4h5b1wYsVVvIjThKZpJL7XlnUsT?usp=sharing)
- In `<PROJECT_ROOT>/vision/hopper_config_extra.py`, assign the `DATA_PATH` and `PROJECT_PATH` variables respectively to the dataset path and to `<PROJECT_ROOT>` (`_base_` can be changed to point to `default_runtime.py` if the MMPose repo was cloned in a custom folder)
- Run `python <PROJECT_ROOT>/vision/mmpose/tools/train.py <PROJECT_ROOT>/vision/hopper_config_extra.py`

(Training scripts for Google Colab environments are available in the `colab` folder, though they will need some tinkering)

The output folder specified by `--output_dir`, contains checkpoints, plots, training logs and test results. Hyperparameters are summarized in `params.txt`.
