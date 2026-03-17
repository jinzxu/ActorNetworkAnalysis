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


# make one random graph by shuffling edges (keep same degrees)
def generate_single_null(Gc, seed):
    H = Gc.copy()
    m = H.number_of_edges()
    nswap = max(1, int(SWAPS_PER_EDGE_MULTIPLIER * m))
    max_tries = nswap * MAX_TRIES_MULTIPLIER
    try:
        nx.double_edge_swap(H, nswap=nswap, max_tries=max_tries, seed=seed)
        return H
    except nx.NetworkXAlgorithmError:
        try:
            nx.double_edge_swap(H, nswap=nswap // 2, max_tries=max_tries, seed=seed + 999999)
            return H
        except:
            return None


# build many random graphs and record their stats as a baseline
def create_null_model(G, seed=42):
    print(f"\nCreating null model ensemble ({NULL_MODEL_ITERATIONS} independent networks)...")
    Gc = get_giant_component(G)

    null_clustering = []
    null_path_length = []
    successful_iterations = 0
    failed_iterations = 0
    start_time = time.time()

    for i in range(NULL_MODEL_ITERATIONS):
        H = generate_single_null(Gc, seed + i)
        if H is None:
            failed_iterations += 1
            continue

        successful_iterations += 1
        clustering = nx.transitivity(H)
        null_clustering.append(clustering)
        path_len = estimate_path_length(H, NULL_MODEL_SAMPLE_SIZE, seed + 1000000 + i)
        null_path_length.append(path_len)

        if (i + 1) % PROGRESS_INTERVAL == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (NULL_MODEL_ITERATIONS - i - 1)
            success_rate = successful_iterations / (i + 1) * 100
            print(f"  [{i+1:4d}/{NULL_MODEL_ITERATIONS}] Success: {success_rate:5.1f}% | "
                  f"Time: {avg_time:5.2f}s/iter | ETA: {eta:6.1f}s")
        if (i + 1) % PROGRESS_DETAIL_INTERVAL == 0 and len(null_clustering) > 0:
            current_c = np.mean(null_clustering)
            current_l = np.mean(null_path_length) if len(null_path_length) > 0 else 0
            print(f"       Current: C={current_c:.4f}, L={current_l:.4f}")

    total_time = time.time() - start_time
    success_rate = successful_iterations / NULL_MODEL_ITERATIONS
    print(f"\n  Complete: {total_time:.1f}s ({total_time/60:.2f}min)")
    print(f"  Success: {successful_iterations}/{NULL_MODEL_ITERATIONS} ({success_rate*100:.1f}%)")

    if len(null_clustering) == 0:
        raise ValueError("Null model failed completely")

    null_stats = {
        'clustering_mean': float(np.mean(null_clustering)),
        'clustering_std': float(np.std(null_clustering)),
        'clustering_min': float(np.min(null_clustering)),
        'clustering_max': float(np.max(null_clustering)),
        'clustering_samples': null_clustering,
        'path_length_mean': float(np.mean(null_path_length)),
        'path_length_std': float(np.std(null_path_length)),
        'path_length_min': float(np.min(null_path_length)),
        'path_length_max': float(np.max(null_path_length)),
        'path_length_samples': null_path_length,
        'successful_iterations': int(successful_iterations),
        'failed_iterations': int(failed_iterations),
        'success_rate': float(success_rate)
    }
    print(f"  Null C: {null_stats['clustering_mean']:.4f} +/- {null_stats['clustering_std']:.4f}")
    print(f"  Null L: {null_stats['path_length_mean']:.4f} +/- {null_stats['path_length_std']:.4f}")
    return null_stats


# compare real network to random baseline, compute small-world sigma
def compare_to_null(real_stats, null_stats):
    print("\nComparing to null model...")
    comparison = {}
    real_c = real_stats['clustering_coefficient']
    null_c_mean = null_stats['clustering_mean']
    null_c_std = null_stats['clustering_std']
    real_l = real_stats['avg_path_length']
    null_l_mean = null_stats['path_length_mean']
    null_l_std = null_stats['path_length_std']
    comparison['real_clustering'] = float(real_c)
    comparison['null_clustering_mean'] = float(null_c_mean)
    comparison['null_clustering_std'] = float(null_c_std)
    comparison['clustering_z_score'] = float((real_c - null_c_mean) / null_c_std) if null_c_std > 0 else None
    comparison['real_path_length'] = float(real_l)
    comparison['null_path_length_mean'] = float(null_l_mean)
    comparison['null_path_length_std'] = float(null_l_std)
    comparison['path_length_z_score'] = float((real_l - null_l_mean) / null_l_std) if null_l_std > 0 else None
    if real_c > 0 and null_c_mean > 0 and real_l > 0 and null_l_mean > 0:
        gamma = real_c / null_c_mean
        lam = real_l / null_l_mean
        comparison['gamma'] = float(gamma)
        comparison['lambda'] = float(lam)
        comparison['small_world_coefficient'] = float(gamma / lam)
        print(f"  gamma (C_real/C_null): {gamma:.3f}")
        print(f"  lambda (L_real/L_null): {lam:.3f}")
        print(f"  Small-world sigma: {comparison['small_world_coefficient']:.3f}")
        if comparison['small_world_coefficient'] > 1:
            print("  -> Small-world properties detected (sigma > 1)")
    else:
        comparison['small_world_coefficient'] = None
    if comparison['clustering_z_score'] is not None:
        print(f"  Clustering z-score: {comparison['clustering_z_score']:.2f}")
    if comparison['path_length_z_score'] is not None:
        print(f"  Path length z-score: {comparison['path_length_z_score']:.2f}")
    return comparison


# plot real vs null model stats side by side
def plot_null_comparison(real_stats, null_stats, output_dir):
    print("\nCreating null model comparison plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    real_c = real_stats['clustering_coefficient']
    null_c_samples = null_stats['clustering_samples']
    bp1 = ax1.boxplot([null_c_samples], positions=[1], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor='#A8DADC', edgecolor='#003049', linewidth=1.5),
                       medianprops=dict(color='#C1121F', linewidth=2.5),
                       whiskerprops=dict(color='#003049', linewidth=1.5),
                       capprops=dict(color='#003049', linewidth=1.5))
    ax1.plot(1, real_c, 'o', markersize=12, color='#C1121F',
            markeredgewidth=2, markeredgecolor='white', label='Observed')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['Null Model'], fontsize=11, fontweight='bold')
    ax1.set_ylabel('Clustering Coefficient', fontsize=13, fontweight='bold')
    ax1.set_title('Clustering: Real vs Null', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(fontsize=10, framealpha=0.95, edgecolor='gray')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    real_l = real_stats['avg_path_length']
    null_l_samples = null_stats['path_length_samples']
    bp2 = ax2.boxplot([null_l_samples], positions=[1], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor='#A8DADC', edgecolor='#003049', linewidth=1.5),
                       medianprops=dict(color='#C1121F', linewidth=2.5),
                       whiskerprops=dict(color='#003049', linewidth=1.5),
                       capprops=dict(color='#003049', linewidth=1.5))
    ax2.plot(1, real_l, 'o', markersize=12, color='#C1121F',
            markeredgewidth=2, markeredgecolor='white', label='Observed')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Null Model'], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Path Length', fontsize=13, fontweight='bold')
    ax2.set_title('Path Length: Real vs Null', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(fontsize=10, framealpha=0.95, edgecolor='gray')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'null_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {plot_path}")


# compute degree, betweenness, eigenvector centrality for each actor
def centrality_analysis(G, output_dir, seed=42):
    print("\nComputing centrality...")
    Gc = get_giant_component(G)
    degree_cent = dict(Gc.degree())
    sample_size = min(BETWEENNESS_SAMPLE_SIZE, Gc.number_of_nodes())
    print(f"  Betweenness (sampling {sample_size} nodes)...")
    nodes = list(Gc.nodes())
    random.seed(seed)
    sampled_nodes = random.sample(nodes, sample_size)
    betweenness_cent = nx.betweenness_centrality_subset(Gc, sources=sampled_nodes,
                                                         targets=nodes, normalized=True)
    print("  Eigenvector (weighted)...")
    try:
        eigenvector_cent = nx.eigenvector_centrality(Gc, max_iter=1000, weight='weight')
    except:
        eigenvector_cent = {n: 0 for n in Gc.nodes()}
    cent_df = pd.DataFrame({
        'node': list(Gc.nodes()),
        'degree': [degree_cent[n] for n in Gc.nodes()],
        'betweenness': [betweenness_cent.get(n, 0) for n in Gc.nodes()],
        'eigenvector': [eigenvector_cent.get(n, 0) for n in Gc.nodes()]
    })
    cent_df = cent_df.sort_values('degree', ascending=False)
    cent_df.to_csv(os.path.join(output_dir, 'centrality.csv'), index=False)
    print("  Top 5:")
    top_5 = []
    for idx, row in cent_df.head(5).iterrows():
        print(f"    {len(top_5)+1}. {row['node']}: deg={int(row['degree'])}, "
              f"btw={row['betweenness']:.4f}, eig={row['eigenvector']:.4f}")
        top_5.append({
            'node': str(row['node']),
            'degree': int(row['degree']),
            'betweenness': float(row['betweenness']),
            'eigenvector': float(row['eigenvector'])
        })
    return top_5


# collect each actor's movie count, scores, votes, genres, career years
def actor_success_features(df, output_dir, seed=42):
    print("\nComputing actor success features...")
    random.seed(seed)
    np.random.seed(seed)
    actor_cols = [c for c in ["actor_1_name", "actor_2_name", "actor_3_name"]
                  if c in df.columns]
    s = {}
    for i, row in df.iterrows():
        actors = set()
        for c in actor_cols:
            v = norm_str(row.get(c, None))
            if v:
                actors.add(v)
        if not actors:
            continue
        score = row.get("imdb_score", np.nan)
        votes = row.get("num_voted_users", np.nan)
        year = row.get("title_year", np.nan)
        genres = row.get("genres_list", [])
        for a in actors:
            if a not in s:
                s[a] = {
                    "n_movies": 0,
                    "imdb_sum": 0.0, "imdb_cnt": 0, "imdb_max": -1e9,
                    "votes_sum": 0.0, "votes_cnt": 0, "votes_max": -1e9,
                    "genres": set(),
                    "ymin": 1e9, "ymax": -1e9
                }
            r = s[a]
            r["n_movies"] += 1
            if pd.notna(score):
                r["imdb_sum"] += float(score)
                r["imdb_cnt"] += 1
                r["imdb_max"] = max(r["imdb_max"], float(score))
            if pd.notna(votes):
                r["votes_sum"] += float(votes)
                r["votes_cnt"] += 1
                r["votes_max"] = max(r["votes_max"], float(votes))
            if pd.notna(year):
                y = float(year)
                r["ymin"] = min(r["ymin"], y)
                r["ymax"] = max(r["ymax"], y)
            for g in genres:
                g2 = norm_str(g)
                if g2:
                    r["genres"].add(g2)
    actors = list(s.keys())
    out = pd.DataFrame({
        "actor": actors,
        "n_movies": [s[a]["n_movies"] for a in actors],
        "imdb_mean": [
            s[a]["imdb_sum"] / s[a]["imdb_cnt"] if s[a]["imdb_cnt"] > 0 else np.nan
            for a in actors
        ],
        "imdb_max": [
            s[a]["imdb_max"] if s[a]["imdb_max"] > -1e8 else np.nan
            for a in actors
        ],
        "votes_mean": [
            s[a]["votes_sum"] / s[a]["votes_cnt"] if s[a]["votes_cnt"] > 0 else np.nan
            for a in actors
        ],
        "votes_max": [
            s[a]["votes_max"] if s[a]["votes_max"] > -1e8 else np.nan
            for a in actors
        ],
        "genre_span": [len(s[a]["genres"]) for a in actors],
        "career_span_years": [
            s[a]["ymax"] - s[a]["ymin"] if (s[a]["ymin"] < 1e8 and s[a]["ymax"] > -1e8)
            else np.nan for a in actors
        ]
    })
    path = os.path.join(output_dir, "actor_success_features.csv")
    out.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return out


# join network stats with actor success data and make scatter plots
def merge_success(G, success_df, centrality_csv_path, output_dir):
    print("\nMerging network position with success features...")
    Gc = get_giant_component(G)
    deg = dict(Gc.degree())
    strength = dict(Gc.degree(weight="weight"))
    df_net = pd.DataFrame({
        "actor": list(Gc.nodes()),
        "degree": [deg.get(a, 0) for a in Gc.nodes()],
        "strength": [strength.get(a, 0.0) for a in Gc.nodes()]
    })
    merged = df_net.merge(success_df, on="actor", how="left")

    # Load centrality data and merge betweenness and eigenvector
    cent_df = pd.read_csv(centrality_csv_path)
    cent_df = cent_df.rename(columns={'node': 'actor'})
    merged = merged.merge(cent_df[['actor', 'betweenness', 'eigenvector']],
                          on='actor', how='left')

    path = os.path.join(output_dir, "network_success_merged.csv")
    merged.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Plot 1: Degree vs IMDb Score
    fig, ax = plt.subplots(figsize=(10, 7))
    x = merged["degree"].to_numpy(dtype=float)
    y = merged["imdb_mean"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=12, alpha=0.5, color='#2E86AB',
              edgecolors='white', linewidths=0.3)
    ax.set_xlabel("Actor Degree (number of collaborations)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean IMDb Score", fontsize=13, fontweight='bold')
    ax.set_title("Network Position vs Success", fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if np.sum(mask) > 10:
        corr, pval = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {pval:.2e}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFF3B0', alpha=0.9, edgecolor='gray'),
                verticalalignment='top')
    fig_path = os.path.join(output_dir, "degree_vs_imdb_score.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Plot 2: Betweenness Centrality vs IMDb Score
    fig, ax = plt.subplots(figsize=(10, 7))
    x = merged["betweenness"].to_numpy(dtype=float)
    y = merged["imdb_mean"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=12, alpha=0.5, color='#E76F51',
              edgecolors='white', linewidths=0.3)
    ax.set_xlabel("Betweenness Centrality", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean IMDb Score", fontsize=13, fontweight='bold')
    ax.set_title("Betweenness Centrality vs Success", fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if np.sum(mask) > 10:
        corr, pval = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {pval:.2e}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFF3B0', alpha=0.9, edgecolor='gray'),
                verticalalignment='top')
    fig_path = os.path.join(output_dir, "betweenness_vs_imdb_score.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Plot 3: Eigenvector Centrality vs IMDb Score
    fig, ax = plt.subplots(figsize=(10, 7))
    x = merged["eigenvector"].to_numpy(dtype=float)
    y = merged["imdb_mean"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=12, alpha=0.5, color='#2A9D8F',
              edgecolors='white', linewidths=0.3)
    ax.set_xlabel("Eigenvector Centrality (weighted)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Mean IMDb Score", fontsize=13, fontweight='bold')
    ax.set_title("Eigenvector Centrality vs Success", fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if np.sum(mask) > 10:
        corr, pval = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {pval:.2e}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFF3B0', alpha=0.9, edgecolor='gray'),
                verticalalignment='top')
    fig_path = os.path.join(output_dir, "eigenvector_vs_imdb_score.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Plot 4: Genre Diversity vs Network Position
    fig, ax = plt.subplots(figsize=(10, 7))
    x = merged["genre_span"].to_numpy(dtype=float)
    y = merged["degree"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], s=12, alpha=0.5, color='#A23B72',
              edgecolors='white', linewidths=0.3)
    ax.set_xlabel("Genre Span (number of unique genres)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Actor Degree", fontsize=13, fontweight='bold')
    ax.set_title("Genre Diversity vs Network Position", fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if np.sum(mask) > 10:
        corr, pval = stats.pearsonr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {pval:.2e}',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#FFF3B0', alpha=0.9, edgecolor='gray'),
                verticalalignment='top')
    fig_path = os.path.join(output_dir, "genre_span_vs_degree.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")


# main entry point: run all steps and save results
def main():
    parser = argparse.ArgumentParser(description='Actor Network Analysis')
    parser.add_argument('--input', type=str, default='movie_metadata.csv')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("="*70)
    print("ACTOR COLLABORATION NETWORK ANALYSIS")
    print("="*70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    start_time = time.time()

    df = read_data(args.input)
    df = clean_data(df)
    G = build_network(df, args.output, seed=args.seed)
    basic_stats = compute_basic_stats(G, seed=args.seed)
    powerlaw_fit = degree_distribution(G, args.output)
    clustering_vs_degree(G, args.output)
    null_stats = create_null_model(G, seed=args.seed)
    null_comparison = compare_to_null(basic_stats, null_stats)
    plot_null_comparison(basic_stats, null_stats, args.output)
    top_central = centrality_analysis(G, args.output, seed=args.seed)
    success_features = actor_success_features(df, args.output, seed=args.seed)
    centrality_csv_path = os.path.join(args.output, 'centrality.csv')
    merge_success(G, success_features, centrality_csv_path, args.output)

    null_stats_clean = {k: v for k, v in null_stats.items()
                        if k not in ['clustering_samples', 'path_length_samples']}

    results = {
        'methodology': 'Clustering: unweighted topology. Weights: eigenvector centrality. Null model: independent degree-preserving randomization ensemble. Community detection: performed in Gephi using Louvain algorithm.',
        'basic_statistics': basic_stats,
        'powerlaw_fit': powerlaw_fit,
        'null_model': null_stats_clean,
        'null_comparison': null_comparison,
        'top_5_central': top_central
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Time: {total_time:.1f}s ({total_time/60:.2f}min)")
    print(f"\nKey Findings:")
    print(f"  Nodes: {basic_stats['num_nodes']:,}")
    print(f"  Edges: {basic_stats['num_edges']:,}")
    print(f"  Avg degree: {basic_stats['avg_degree']:.2f}")
    print(f"  Clustering: {basic_stats['clustering_coefficient']:.4f}")
    print(f"  Assortativity: {basic_stats['degree_assortativity']:.3f}")
    if powerlaw_fit.get('alpha'):
        print(f"  Power-law alpha: {powerlaw_fit['alpha']:.3f} (xmin={powerlaw_fit['xmin']})")
        if powerlaw_fit['vs_lognormal']['p'] < 0.05:
            winner = "power_law" if powerlaw_fit['vs_lognormal']['R'] > 0 else "lognormal"
            print(f"  Distribution comparison: {winner} preferred (p={powerlaw_fit['vs_lognormal']['p']:.4f})")
        else:
            print(f"  Distribution comparison: inconclusive (p={powerlaw_fit['vs_lognormal']['p']:.4f})")
    if null_comparison.get('small_world_coefficient'):
        print(f"  Small-world sigma: {null_comparison['small_world_coefficient']:.3f}")
    print(f"  Null success rate: {null_stats['success_rate']*100:.1f}%")
    print(f"\n=== OUTPUT FILES ===")
    print(f"Saved to: {args.output}/")
    print("  gephi_nodes.csv & gephi_edges.csv (full network for Gephi)")
    for t in LABEL_DEGREE_THRESHOLDS:
        print(f"  gephi_nodes_degree{t}_labels.csv & gephi_edges_degree{t}_labels.csv (labels for degree >= {t})")
    print("  degree_distribution_pdf.png & ccdf.png (with power-law fit)")
    print("  degree_distribution_pdf_raw.png & ccdf_raw.png (without fit)")
    print("  clustering_vs_degree.png")
    print("  null_comparison.png")
    print("  degree_vs_imdb_score.png")
    print("  betweenness_vs_imdb_score.png")
    print("  eigenvector_vs_imdb_score.png")
    print("  genre_span_vs_degree.png")
    print("  centrality.csv")
    print("  All CSV and JSON files")
    print("="*70)


if __name__ == '__main__':
    main()