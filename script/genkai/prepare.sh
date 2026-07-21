#!/bin/bash
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j

module load cuda

uv run prepare.py