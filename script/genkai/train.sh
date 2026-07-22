#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -L jobenv=singularity
#PJM -j

# module load cuda
module load singularity-ce

# Singularityイメージ
IMAGE=/home/pj24001974/ku50001532/nlp-singularity/nlp-singularity.sif

# プロジェクトディレクトリ
WORKDIR=/home/pj24001974/ku50001532/projects/autoresearch

singularity exec \
    --nv \
    --bind ${WORKDIR}:${WORKDIR} \
    --pwd ${WORKDIR} \
    ${IMAGE} \
    bash -lc "uv run train.py"
