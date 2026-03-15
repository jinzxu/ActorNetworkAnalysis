import subprocess
import importlib

REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'networkx': 'networkx',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'powerlaw': 'powerlaw',
}

# install missing packages
def check_and_install():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [__import__('sys').executable, '-m', 'pip', 'install', '--quiet'] + missing
        )
        print("Installation complete.")

check_and_install()

import os
import sys
import time
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import powerlaw
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

PATH_SAMPLE_SIZE = 200
BETWEENNESS_SAMPLE_SIZE = 500
NULL_MODEL_ITERATIONS = 1000
NULL_MODEL_SAMPLE_SIZE = 200
SWAPS_PER_EDGE_MULTIPLIER = 10
MAX_TRIES_MULTIPLIER = 20
PROGRESS_INTERVAL = 10
PROGRESS_DETAIL_INTERVAL = 100
CLUSTERING_DEGREE_BINS = 20
LABEL_DEGREE_THRESHOLDS = [15, 20]

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3