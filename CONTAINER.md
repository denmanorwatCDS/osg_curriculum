docker build -t dvasilev:girol
docker run -it --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --ipc=host <image_id>
python smart_ddqn/main.py --config smart_ddqn/configs/chair.json