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

# get the biggest connected part of the graph
def get_giant_component(G):
    if G.number_of_nodes() == 0:
        return G
    components = list(nx.connected_components(G))
    if len(components) == 0:
        return G
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


# estimate average shortest path by sampling some nodes
def estimate_path_length(G, sample_size, seed):
    nodes = list(G.nodes())
    k = min(sample_size, len(nodes))
    rng = random.Random(seed)
    sampled_nodes = rng.sample(nodes, k)
    total_distance = 0
    total_pairs = 0
    for node in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(G, node)
        for target, dist in lengths.items():
            if target != node:
                total_distance += dist
                total_pairs += 1
    return float(total_distance / total_pairs) if total_pairs > 0 else 0.0


# compute basic graph stats like degree, clustering, path length
def compute_basic_stats(G, seed=42):
    print("\nComputing basic statistics...")
    s = {}
    s['num_nodes'] = int(G.number_of_nodes())
    s['num_edges'] = int(G.number_of_edges())
    s['num_components'] = int(nx.number_connected_components(G))
    Gc = get_giant_component(G)
    s['giant_component_size'] = int(Gc.number_of_nodes())
    degrees = [d for n, d in Gc.degree()]
    s['avg_degree'] = float(np.mean(degrees))
    s['max_degree'] = int(np.max(degrees))
    s['min_degree'] = int(np.min(degrees))
    s['median_degree'] = float(np.median(degrees))
    print("Computing clustering coefficient...")
    s['clustering_coefficient'] = float(nx.transitivity(Gc))
    s['avg_clustering'] = float(nx.average_clustering(Gc))
    print("Computing degree assortativity...")
    s['degree_assortativity'] = float(nx.degree_assortativity_coefficient(Gc))
    print(f"Estimating average path length (sampling {PATH_SAMPLE_SIZE} nodes)...")
    s['avg_path_length'] = estimate_path_length(Gc, PATH_SAMPLE_SIZE, seed)
    if Gc.number_of_nodes() < 1000:
        print("Computing diameter...")
        s['diameter'] = int(nx.diameter(Gc))
    else:
        print("Network too large, skipping diameter computation")
        s['diameter'] = None
    print(f"  Nodes: {s['num_nodes']}")
    print(f"  Edges: {s['num_edges']}")
    print(f"  Avg degree: {s['avg_degree']:.2f}")
    print(f"  Clustering: {s['clustering_coefficient']:.4f}")
    print(f"  Assortativity: {s['degree_assortativity']:.4f}")
    print(f"  Avg path length: {s['avg_path_length']:.4f}")
    return s

# fit power-law to degree distribution and save plots
def degree_distribution(G, output_dir):
    print("\nAnalyzing degree distribution with powerlaw library...")
    os.makedirs(output_dir, exist_ok=True)
    Gc = get_giant_component(G)
    if Gc.number_of_nodes() == 0:
        print("  Warning: Giant component is empty")
        return {}
    degrees = np.array([d for _, d in Gc.degree()])
    if len(degrees) == 0:
        print("  Warning: No degrees available")
        return {}

    degree_counts = pd.Series(degrees).value_counts().sort_index()
    degree_df = pd.DataFrame({
        "degree": degree_counts.index.astype(int),
        "count": degree_counts.values.astype(int)
    })
    csv_path = os.path.join(output_dir, "degree_distribution.csv")
    degree_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
    alpha = float(fit.power_law.alpha)
    xmin = int(fit.power_law.xmin)
    sigma = float(fit.power_law.sigma)
    n_tail = int(np.sum(degrees >= xmin))

    R_lognormal, p_lognormal = fit.distribution_compare('power_law', 'lognormal')
    R_exponential, p_exponential = fit.distribution_compare('power_law', 'exponential')
    R_stretched, p_stretched = fit.distribution_compare('power_law', 'stretched_exponential')

    powerlaw_result = {
        'alpha': alpha,
        'xmin': xmin,
        'sigma': sigma,
        'n_tail': n_tail,
        'vs_lognormal': {'R': float(R_lognormal), 'p': float(p_lognormal)},
        'vs_exponential': {'R': float(R_exponential), 'p': float(p_exponential)},
        'vs_stretched_exponential': {'R': float(R_stretched), 'p': float(p_stretched)}
    }

    print(f"  Power-law: alpha={alpha:.3f}, xmin={xmin}, sigma={sigma:.4f}")
    print(f"  Tail size: {n_tail} nodes")
    print(f"  vs Lognormal:    R={R_lognormal:+.4f}, p={p_lognormal:.4f}")
    print(f"  vs Exponential:  R={R_exponential:+.4f}, p={p_exponential:.4f}")
    print(f"  vs Stretched Exp: R={R_stretched:+.4f}, p={p_stretched:.4f}")

    max_degree = int(degree_df["degree"].max())
    unique_degrees = np.sort(np.unique(degrees))
    ccdf = np.array([np.sum(degrees >= k) / len(degrees) for k in unique_degrees])

    # --- PDF with power-law fit ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(degree_df["degree"], degree_df["count"], "o",
              markersize=7, alpha=0.7, color='#2E86AB', markeredgewidth=0.5,
              markeredgecolor='white', label="Observed")
    if max_degree >= xmin:
        k_fit = np.arange(xmin, max_degree + 1)
        tail_total_count = degree_df[degree_df['degree'] >= xmin]['count'].sum()
        power_law_line = tail_total_count * (alpha - 1) / xmin * (k_fit / xmin) ** (-alpha)
        ax.loglog(k_fit, power_law_line, "-", linewidth=3.0, color='#C1121F',
                  label=f"Power-law fit (α={alpha:.2f}, x_min={xmin})")
        ax.axvline(xmin, color='#F77F00', linestyle='--', linewidth=2.0,
                  alpha=0.8, label=f'x_min={xmin}')
    ax.set_xlabel("Degree (k)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count", fontsize=13, fontweight='bold')
    ax.set_title("Degree Distribution with Power-law Fit", fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "degree_distribution_pdf.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {plot_path}")

    # --- PDF without fit (raw data only) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(degree_df["degree"], degree_df["count"], "o",
              markersize=7, alpha=0.7, color='#2E86AB', markeredgewidth=0.5,
              markeredgecolor='white', label="Observed")
    ax.set_xlabel("Degree (k)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Count", fontsize=13, fontweight='bold')
    ax.set_title("Degree Distribution", fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "degree_distribution_pdf_raw.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {plot_path}")

    # --- CCDF with power-law fit ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(unique_degrees, ccdf, "o", markersize=7, alpha=0.7,
              color='#2E86AB', markeredgewidth=0.5, markeredgecolor='white',
              label="Observed CCDF")
    k_tail = unique_degrees[unique_degrees >= xmin]
    if len(k_tail) > 0:
        ccdf_tail = (k_tail / float(xmin)) ** (1 - alpha)
        scale_idx = np.where(unique_degrees == xmin)[0]
        scale_factor = ccdf[scale_idx[0]] if len(scale_idx) > 0 else 1.0
        ccdf_fit = scale_factor * ccdf_tail
        ax.loglog(k_tail, ccdf_fit, "-", linewidth=3.0, color='#C1121F',
                  label=f"Power-law fit (α={alpha:.2f})")
    ax.set_xlabel("Degree (k)", fontsize=13, fontweight='bold')
    ax.set_ylabel("P(K ≥ k)", fontsize=13, fontweight='bold')
    ax.set_title("Complementary Cumulative Distribution (CCDF)", fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "degree_distribution_ccdf.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {plot_path}")

    # --- CCDF without fit (raw data only) ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(unique_degrees, ccdf, "o", markersize=7, alpha=0.7,
              color='#2E86AB', markeredgewidth=0.5, markeredgecolor='white',
              label="Observed CCDF")
    ax.set_xlabel("Degree (k)", fontsize=13, fontweight='bold')
    ax.set_ylabel("P(K ≥ k)", fontsize=13, fontweight='bold')
    ax.set_title("Complementary Cumulative Distribution (CCDF)", fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "degree_distribution_ccdf_raw.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {plot_path}")

    return powerlaw_result


# plot how clustering changes with node degree
def clustering_vs_degree(G, output_dir):
    print("\nAnalyzing clustering vs degree...")
    Gc = get_giant_component(G)
    clustering = nx.clustering(Gc)
    degrees = dict(Gc.degree())
    data = []
    for node in Gc.nodes():
        data.append({
            'degree': degrees[node],
            'clustering': clustering[node]
        })
    df = pd.DataFrame(data)
    df_binned = df[df['degree'] > 0].copy()
    if len(df_binned) == 0:
        print("  Warning: No valid data for clustering vs degree")
        return
    min_deg = df_binned['degree'].min()
    max_deg = df_binned['degree'].max()
    if max_deg > min_deg:
        bins = np.logspace(np.log10(min_deg), np.log10(max_deg), CLUSTERING_DEGREE_BINS)
        df_binned['degree_bin'] = pd.cut(df_binned['degree'], bins=bins, include_lowest=True)
        grouped = df_binned.groupby('degree_bin', observed=True).agg({
            'degree': 'mean',
            'clustering': 'mean'
        }).dropna()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.loglog(df['degree'], df['clustering'], 'o', alpha=0.15,
                  markersize=4, color='lightblue', label='Individual nodes')
        ax.loglog(grouped['degree'], grouped['clustering'], 'o-',
                  markersize=10, linewidth=2.5, color='#003049',
                  markeredgewidth=1.5, markeredgecolor='white',
                  label='Binned average')
        ax.set_xlabel('Degree (k)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Clustering Coefficient C(k)', fontsize=13, fontweight='bold')
        ax.set_title('Clustering Coefficient vs Degree', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, framealpha=0.95, edgecolor='gray', fancybox=True)
        ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'clustering_vs_degree.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {plot_path}")
    else:
        print("  Warning: Insufficient degree range for clustering analysis")
