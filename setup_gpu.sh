#!/bin/bash
set -e

# Self-contained setup for GPU machine (Linux + NVIDIA GPU)

# 1. Create isolated conda env
conda create -n caramel python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate caramel

# 2. Install Python deps with GPU TensorFlow
pip install 'tensorflow[and-cuda]' scikit-learn keras numpy matplotlib

# 3. Verify GPU
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'GPUs: {gpus}'); assert gpus, 'No GPU found!'"

# 4. Build C++ binary
cd deps/LearnedStaticFunction
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTFLITE_ENABLE_XNNPACK=OFF
cmake --build . --target ribbon_learned_bench -j$(nproc)
cd ../../..

# 5. Verify
./deps/LearnedStaticFunction/build/ribbon_learned_bench --help
echo "Setup complete!"
