FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ADD . /osg_girol
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential software-properties-common git curl libbz2-dev libffi-dev liblzma-dev \
      libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev \
      libxml2-dev libxmlsec1-dev llvm make tk-dev wget \
      xz-utils zlib1g-dev nano rsync vim tree unzip htop tmux xvfb patchelf ca-certificates \
      bash-completion libjpeg-dev libpng-dev \
      ffmpeg cmake swig libssl-dev libcurl4-openssl-dev libopenmpi-dev python3-dev zlib1g-dev \
      qtbase5-dev qtdeclarative5-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 \
      libgl1 libglx-mesa0 libosmesa6 libosmesa6-dev libglew-dev mesa-utils libglew-dev libc-dev \
      libgl1-mesa-glx libglfw3 python3-setuptools libxrandr-dev libxinerama-dev libxcursor-dev \
      libxi-dev
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /osg_girol
RUN uv venv .osg --python 3.9
ENV PATH="/osg_girol/.osg/bin:$PATH"

RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN git submodule update --init Grounded-Segment-Anything 
WORKDIR Grounded-Segment-Anything
RUN git submodule update --init Tag2Text
WORKDIR ..
RUN git submodule update --init home-robot

RUN uv pip install "spacy==3.7.6"
RUN uv pip install salesforce-lavis openai
RUN CUDA_VISIBLE_DEVICES="" uv pip install -e Grounded-Segment-Anything/GroundingDINO --no-build-isolation
RUN uv pip install -r Grounded-Segment-Anything/Tag2Text/requirements.txt --no-build-isolation

RUN uv pip install transformers==4.26.1
RUN CUDA_VISIBLE_DEVICES="" uv pip install torch-cluster torch-geometric --no-build-isolation

RUN git clone --branch v0.2.5 --depth 1 https://github.com/facebookresearch/habitat-sim.git
WORKDIR habitat-sim
RUN uv pip install -r requirements.txt

RUN uv pip install --upgrade pip setuptools wheel
RUN uv run /osg_girol/.osg/bin/python setup.py install --bullet
WORKDIR ..

RUN uv pip install -e home-robot/src/home_robot --no-build-isolation
WORKDIR home-robot
RUN git submodule update --init --recursive src/third_party/habitat-lab
WORKDIR ..
RUN uv pip install -e home-robot/src/third_party/habitat-lab/habitat-lab --no-build-isolation
RUN uv pip install -e home-robot/src/third_party/habitat-lab/habitat-baselines --no-build-isolation
RUN uv pip install -e home-robot/src/home_robot_sim --no-build-isolation
RUN CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
RUN uv pip install scikit-fmm sophuspy

# Maybe delete it?

RUN echo "from sophuspy import *" > "$(python -c 'import site; print(site.getsitepackages()[0])')/sophus.py"
RUN uv pip install "numpy<2.0.0"
RUN uv pip install "skrl==1.4.3"
RUN uv pip install "comet_ml==3.58.4"

RUN mkdir -p checkpoints logs data/scene_datasets
RUN wget -P checkpoints https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
RUN wget -P checkpoints https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN uv tool install gdown
WORKDIR ./data/scene_datasets
RUN gdown --fuzzy "https://drive.google.com/file/d/1nnp15jI94yt1hCK_9D1FN9JaY4SFblr4/view?usp=sharing" -O archive.zip
RUN unzip archive.zip
RUN mv data/scene_datasets/hm3d  .
RUN mv data/scene_datasets/hm3d_v0.2  .
RUN rm -rf data/
WORKDIR ../.. 

ENV __NV_PRIME_RENDER_OFFLOAD=1
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
# habitat-sim needs the EGL/GL libs, which the toolkit only injects with the graphics capability
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
