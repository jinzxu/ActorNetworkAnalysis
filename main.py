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


# clean a string, return None if blank or NaN
def norm_str(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = " ".join(s.split())
    return s


# split "Action|Drama|..." into a list
def parse_genres(x):
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split("|")]
    return [p for p in parts if p]


# read the CSV file and return a dataframe
def read_data(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path, low_memory=False)
        if len(df) == 0:
            raise ValueError("Dataset is empty")
        print(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


# fix column types and remove bad rows
def clean_data(df):
    required_cols = ['movie_title', 'actor_1_name', 'actor_2_name', 'actor_3_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    df = df.copy()
    if "movie_title" in df.columns:
        df["movie_title"] = df["movie_title"].apply(norm_str)
    for c in ["actor_1_name", "actor_2_name", "actor_3_name"]:
        if c in df.columns:
            df[c] = df[c].apply(norm_str)
    for c in ["title_year", "imdb_score", "num_voted_users"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(parse_genres)
    else:
        df["genres_list"] = [[] for _ in range(len(df))]
    if "movie_title" in df.columns:
        df = df[df["movie_title"].notna()].reset_index(drop=True)
    print(f"Cleaned data: {len(df)} rows")
    return df


# build a graph where actors are nodes and shared movies are edges
def build_network(df, output_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    print("Building network...")
    actor_cols = [c for c in ["actor_1_name", "actor_2_name", "actor_3_name"]
                  if c in df.columns]
    w = {}
    credits = {}
    for idx, row in df.iterrows():
        actors = []
        for c in actor_cols:
            v = norm_str(row.get(c, None))
            if v:
                actors.append(v)
        actors = sorted(set(actors))
        if len(actors) == 0:
            pass
        elif len(actors) == 1:
            a = actors[0]
            credits[a] = credits.get(a, 0) + 1
        else:
            for a in actors:
                credits[a] = credits.get(a, 0) + 1
            for x in range(len(actors)):
                for y in range(x + 1, len(actors)):
                    u, v = actors[x], actors[y]
                    if u <= v:
                        k = (u, v)
                    else:
                        k = (v, u)
                    w[k] = w.get(k, 0) + 1
    G = nx.Graph()
    for a, c in credits.items():
        G.add_node(a, movies_credited=int(c))
    for (u, v), ww in w.items():
        G.add_edge(u, v, weight=int(ww))
    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    save_network_files(G, output_dir)
    return G


# save node and edge CSV files for Gephi
def save_network_files(G, output_dir):
    print("Saving network files for Gephi...")
    nodes_list = list(G.nodes())
    degrees = dict(G.degree())
    strength = dict(G.degree(weight="weight"))

    # Full network CSV (all nodes with labels)
    nodes_df = pd.DataFrame({
        "Id": nodes_list,
        "Label": nodes_list,
        "movies_credited": [G.nodes[n].get("movies_credited", 0) for n in nodes_list],
        "degree": [degrees[n] for n in nodes_list],
        "weighted_degree": [strength[n] for n in nodes_list]
    })
    nodes_df.to_csv(os.path.join(output_dir, "gephi_nodes.csv"), index=False)

    edges_df = pd.DataFrame({
        "Source": [u for u, v in G.edges()],
        "Target": [v for u, v in G.edges()],
        "Weight": [G[u][v].get("weight", 1) for u, v in G.edges()],
        "Type": ["Undirected"] * G.number_of_edges()
    })
    edges_df.to_csv(os.path.join(output_dir, "gephi_edges.csv"), index=False)
    print(f"  Saved: gephi_nodes.csv ({len(nodes_df)} nodes)")
    print(f"  Saved: gephi_edges.csv ({len(edges_df)} edges)")

    # Filtered CSVs: generate node/edge files for each degree threshold
    for threshold in LABEL_DEGREE_THRESHOLDS:
        nodes_thresh_df = pd.DataFrame({
            "Id": nodes_list,
            "Label": [n if degrees[n] >= threshold else "" for n in nodes_list],
            "movies_credited": [G.nodes[n].get("movies_credited", 0) for n in nodes_list],
            "degree": [degrees[n] for n in nodes_list],
            "weighted_degree": [strength[n] for n in nodes_list]
        })
        nodes_fname = f"gephi_nodes_degree{threshold}_labels.csv"
        nodes_thresh_df.to_csv(os.path.join(output_dir, nodes_fname), index=False)

        edges_fname = f"gephi_edges_degree{threshold}_labels.csv"
        edges_df.to_csv(os.path.join(output_dir, edges_fname), index=False)

        labeled_count = sum(1 for n in nodes_list if degrees[n] >= threshold)
        print(f"  Saved: {nodes_fname} ({labeled_count} nodes with labels, degree >= {threshold})")
        print(f"  Saved: {edges_fname} ({len(edges_df)} edges, full network)")

