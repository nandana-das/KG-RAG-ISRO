"""Generate publication-quality charts from evaluation results."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "data" / "results" / "baseline_results.json"
ABLATION_PATH = ROOT / "data" / "results" / "ablation_results.json"
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "isro_qa.json"
OUTPUT_DIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

NAVY  = '#012258'
RED   = '#C00000'
GREEN = '#375623'


def plot_main_results():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    systems = ['KG-RAG\n(ours)', 'BM25\n+ LLM', 'Vanilla\nRAG']
    x = np.arange(len(systems))
    colors = [NAVY, RED, GREEN]

    rouge = [0.274, 0.295, 0.269]
    bars1 = axes[0].bar(x, rouge, color=colors, width=0.5, edgecolor='white')
    axes[0].set_ylim(0, 0.40)
    axes[0].set_ylabel('Score')
    axes[0].set_title('(a) ROUGE-L', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(systems)
    for bar, val in zip(bars1, rouge):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    cov = [0.403, 0.438, 0.399]
    bars2 = axes[1].bar(x, cov, color=colors, width=0.5, edgecolor='white')
    axes[1].set_ylim(0, 0.58)
    axes[1].set_ylabel('Score')
    axes[1].set_title('(b) Answer Coverage', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(systems)
    for bar, val in zip(bars2, cov):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    fig.suptitle('Performance Comparison on ISRO-QA (200 Questions)', fontweight='bold', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / 'results_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')


def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    configs = ['KG-only', 'FAISS-only', 'Full\nKG-RAG']
    x = np.arange(len(configs))
    colors = ['#AAAAAA', '#5B9BD5', NAVY]

    abl_rouge = [0.093, 0.283, 0.295]
    bars3 = axes[0].bar(x, abl_rouge, color=colors, width=0.5, edgecolor='white')
    axes[0].set_ylim(0, 0.40)
    axes[0].set_ylabel('Score')
    axes[0].set_title('(a) ROUGE-L', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    for bar, val in zip(bars3, abl_rouge):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    abl_cov = [0.108, 0.408, 0.456]
    bars4 = axes[1].bar(x, abl_cov, color=colors, width=0.5, edgecolor='white')
    axes[1].set_ylim(0, 0.60)
    axes[1].set_ylabel('Score')
    axes[1].set_title('(b) Answer Coverage', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs)
    for bar, val in zip(bars4, abl_cov):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    axes[1].annotate('', xy=(2, abl_cov[2]), xytext=(1, abl_cov[1]),
        arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.5))
    axes[1].text(1.55, (abl_cov[1]+abl_cov[2])/2 + 0.02, '+4.9pp',
                 ha='center', fontsize=7.5, color=NAVY, fontweight='bold')

    fig.suptitle('Ablation Study: KG Contribution over Dense Retrieval (50-Question Sample)',
                 fontweight='bold', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / 'ablation_results.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')


def plot_idk_per_tier():
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    tiers = ['Tier 1: Factoid\n(100 Qs)', 'Tier 2: Multi-hop\n(60 Qs)', 'Tier 3: Timeline\n(40 Qs)']
    kgrag_idk   = [14.0, 6.7,  2.5]
    bm25_idk    = [4.0,  3.3,  0.0]
    vanilla_idk = [13.0, 11.7, 15.0]

    x = np.arange(len(tiers))
    w = 0.25

    b1 = ax.bar(x - w, kgrag_idk,   w, label='KG-RAG (ours)', color=NAVY,  edgecolor='white')
    b2 = ax.bar(x,     bm25_idk,    w, label='BM25 + LLM',    color=RED,   edgecolor='white')
    b3 = ax.bar(x + w, vanilla_idk, w, label='Vanilla RAG',   color=GREEN, edgecolor='white')

    ax.set_ylabel('IDK Rate (%)')
    ax.set_title('IDK Rate per Difficulty Tier', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 20)
    ax.legend(loc='upper right')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=7)

    ax.annotate('Best\n(2.5%)', xy=(x[2]-w, 2.5), xytext=(x[2]-w-0.5, 10),
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.2),
                fontsize=7, color=NAVY, fontweight='bold', ha='center')

    plt.tight_layout()
    out = OUTPUT_DIR / 'idk_per_tier.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {out}')


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_main_results()
    plot_ablation()
    plot_idk_per_tier()
    print('All charts generated in paper/figures/')