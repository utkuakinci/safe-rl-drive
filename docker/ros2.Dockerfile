# ROS2 Humble + C++ policy node (ONNX Runtime backend) + highway-env bridge.
FROM ros:humble

ARG ORT_VERSION=1.20.1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions wget ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz \
    && tar xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ORT_VERSION} /opt/onnxruntime \
    && rm onnxruntime-linux-x64-${ORT_VERSION}.tgz
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Ubuntu 22.04 ships pip 22.0, which cannot read PEP 621 metadata from pyproject.toml (it installs the
# project as "UNKNOWN" without dependencies), so upgrade pip first. numpy<2 keeps the apt-installed ROS
# python packages happy; setuptools<80 + packaging>=24.2 keep colcon's setup.py builds working.
ARG PIN="numpy<2 setuptools<80 packaging>=24.2"
RUN python3 -m pip install --no-cache-dir --upgrade "pip>=24" \
    && python3 -m pip install --no-cache-dir $PIN torch --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
    && python3 -m pip install --no-cache-dir ".[deploy]" $PIN \
    && python3 -c "import safe_rl_drive, highway_env, onnxruntime"

COPY ros2_ws /app/ros2_ws
RUN . /opt/ros/humble/setup.sh && cd /app/ros2_ws \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/opt/onnxruntime

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
# docker run --rm -v $PWD/models:/models srd-ros2 \
#     ros2 launch rl_policy_node drive.launch.py model_path:=/models/rppo_highway.onnx episodes:=20
CMD ["ros2", "launch", "rl_policy_node", "drive.launch.py"]
