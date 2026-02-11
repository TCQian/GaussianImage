#!/bin/bash
#SBATCH --job-name=GaussianImage         # Job name
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL                  # Get email for all status updates
#SBATCH --mail-user=e0407638@u.nus.edu   # Email for notifications

source ~/.bashrc
conda activate gaussianimage
cd gsplat
pip install .[dev]
cd ..

# Define variables for easy updating.
DATA_NAME="Beauty"
MODEL_NAME="GaussianImage_Cholesky"
TRAIN_ITERATIONS=50000
QUANT_ITERATIONS=50000

# Default values for parameters to be overridden.
NUM_POINTS=750
START_FRAME=0
NUM_FRAMES=1

# Parse command-line arguments.
# Usage: ./script.sh --data_name MyData --num_points 30000 --start_frame 40 --num_frames 15
while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--data_name)
            DATA_NAME="$2"
            shift 2
            ;;
        -p|--num_points)
            NUM_POINTS="$2"
            shift 2
            ;;
        -s|--start_frame)
            START_FRAME="$2"
            shift 2
            ;;
        -n|--num_frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: $0 [--data_name <value>] [--num_points <value>] [--start_frame <value>] [--num_frames <value>]"
            exit 1
            ;;
    esac
done

echo "Starting GaussianImage_${DATA_NAME}_${NUM_FRAMES}_${NUM_POINTS}..."

# Define dataset and checkpoint paths using the variables.
YUV_PATH="/home/e/e0407638/github/GaussianVideo/YUV/${DATA_NAME}_1920x1080_120fps_420_8bit_YUV.yuv"
DATASET_PATH="/home/e/e0407638/github/GaussianVideo/dataset/${DATA_NAME}/"
CHECKPOINT_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints/${DATA_NAME}/${MODEL_NAME}_${TRAIN_ITERATIONS}_${NUM_POINTS}/"
CHECKPOINT_QUANT_PATH="/home/e/e0407638/github/GaussianVideo/checkpoints_quant/${DATA_NAME}/${MODEL_NAME}_${QUANT_ITERATIONS}_${NUM_POINTS}/"

python test_image.py --checkpoint ${CHECKPOINT_PATH}/gaussian_model.pth.tar --H 1080 --W 1920
