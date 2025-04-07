ARG IMAGE
FROM ${IMAGE}

# # temporary fix torch version compatibility
# RUN pip uninstall -y torchvision torchaudio \
#     && pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install clearml

ENTRYPOINT [ "" ]

ARG WORKDIR
WORKDIR ${WORKDIR}
