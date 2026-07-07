FROM nvcr.io/nvidia/isaac-lab:2.3.2

ARG UID=1000009
ARG GID=1000009
ARG USERNAME=hvlab

USER root

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    chmod o+rx /isaac-sim && \
    chown ${UID}:${GID} /isaac-sim/kit/cache && \
    chown ${UID}:${GID} /workspace && \
    mkdir -p /home/${USERNAME}/.cache/ov /home/${USERNAME}/.cache/pip \
             /home/${USERNAME}/.cache/nvidia/GLCache /home/${USERNAME}/.nv/ComputeCache \
             /home/${USERNAME}/.nvidia-omniverse/logs /home/${USERNAME}/.local/share/ov/data && \
    chown -R ${UID}:${GID} /home/${USERNAME} && \
    echo "alias isaaclab=/workspace/isaaclab/isaaclab.sh" >> /home/${USERNAME}/.bashrc && \
    echo "alias tensorboard='/workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/_isaac_sim/tensorboard'" >> /home/${USERNAME}/.bashrc && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python3 && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip3 && \
    chmod +x /usr/local/bin/python /usr/local/bin/python3 /usr/local/bin/pip /usr/local/bin/pip3

COPY ASAP /workspace/ASAP
RUN pip install -e /workspace/ASAP -e /workspace/ASAP/isaac_utils && \
    chown -R ${UID}:${GID} /workspace/ASAP

USER ${UID}:${GID}
ENV HOME=/home/${USERNAME}
WORKDIR /workspace

ENTRYPOINT []
CMD ["bash"]
