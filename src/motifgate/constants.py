from __future__ import annotations

"""Global constants, defaults and runtime configuration for MotifGate."""

import argparse
import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
except Exception:
    pass

BASES = "ACGT"

B2I = {b: i for i, b in enumerate(BASES)}

COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}

CGR_BITS = {"A": (0, 0), "T": (1, 0), "G": (1, 1), "C": (0, 1)}

SEED = 251031

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFAC_DIR = "Shared/transfac"

SITES_DIR = "Shared/sites"

JASPAR_PWM_DIR = "Shared/Jaspar_pwms"

FIXED_LEN = 14

TRAIN_FRAC = 0.70

VAL_FRAC = 0.15

TEST_FRAC = 0.15

NEG_PER_POS = 2

NEG_CAP_Q = 0.10

NEG_GC_TOL_FRAC = 0.12

NEG_CANDIDATE_TRIES = 32

BACKGROUND_STRIDE = 3

BACKGROUND_LIMIT = 250000

D_MODEL = 32

N_HEADS = 4

NUM_EXPERTS = 4

DROPOUT = 0.20

EPOCHS = 50

PATIENCE = 5

BATCH_SIZE = 256

LR = 2e-3

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

RC_AUGMENTATION = True

USE_SE_BLOCK = True

WARMUP_EPOCHS_FRAC = 0.10

LR_MIN_FACTOR = 0.01

RESIDUAL_GATE_INIT = 0.1
