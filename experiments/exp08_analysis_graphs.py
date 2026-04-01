"""
Comprehensive analysis and visualization of larger model experiments.
Generates publication-quality graphs and validates all claims.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================================
# Raw data from exp07 runs (corrected — best vanilla = lowest PPL, not first alphabetically)
# ============================================================================

models_data = {
    'gpt2': {
        'params': '124M', 'layers': 12, 'head_dim': 64, 'heads': 12, 'kv_heads': 12,
        'ppl_base': 23.53,
        'results': [
            ('Vanilla K4/V2 w64',    23.86, +1.4, 'vanilla'),
            ('Vanilla K4/V2 w32',    24.11, +2.5, 'vanilla'),
            ('Vanilla K3/V2 w32',    27.99, +19.0, 'vanilla'),
            ('Sink(4)+Heuristic w32', 24.94, +6.0, 'heuristic'),
            ('Sink(8)+Heuristic w32', 24.88, +5.7, 'heuristic'),
            ('Sink(4)+Measured w32',  24.67, +4.8, 'measured'),
            ('Sink(8)+Measured w32',  24.90, +5.8, 'measured'),
            ('Sink(4)+K4/V2 w32',    23.77, +1.0, 'sink_only'),
            ('Sink(8)+K4/V2 w32',    23.80, +1.2, 'sink_only'),
        ],
        'sensitivities': [1.0, 0.052, 0.046, 0.108, 0.028, 0.031, 0.050, 0.021, 0.020, 0.010, 0.015, 0.003],
        'sink_attention': [0.038, 0.111, 0.177, 0.361, 0.413, 0.597, 0.555, 0.662, 0.525, 0.637, 0.603, 0.553],
    },
    'gpt2-medium': {
        'params': '355M', 'layers': 24, 'head_dim': 64, 'heads': 16, 'kv_heads': 16,
        'ppl_base': 18.08,
        'results': [
            ('Vanilla K4/V2 w64',    18.36, +1.6, 'vanilla'),
            ('Vanilla K4/V2 w32',    18.44, +2.0, 'vanilla'),
            ('Vanilla K3/V2 w32',    18.80, +4.0, 'vanilla'),
            ('Sink(4)+Heuristic w32', 18.91, +4.6, 'heuristic'),
            ('Sink(8)+Heuristic w32', 19.12, +5.8, 'heuristic'),
            ('Sink(4)+Measured w32',  18.98, +5.0, 'measured'),
            ('Sink(8)+Measured w32',  19.23, +6.4, 'measured'),
            ('Sink(4)+K4/V2 w32',    18.58, +2.8, 'sink_only'),
            ('Sink(8)+K4/V2 w32',    18.55, +2.6, 'sink_only'),
        ],
        'sensitivities': [1.0, 0.019, 0.009, 0.007, 0.067, 0.019, 0.010, 0.008, 0.005, 0.007, 0.005, 0.010, 0.023, 0.043, 0.011, 0.005, 0.003, 0.007, 0.009, 0.007, 0.001, 0.002, 0.001, 0.0003],
        'sink_attention': [0.061, 0.114, 0.100, 0.110, 0.283, 0.438, 0.628, 0.617, 0.473, 0.664, 0.501, 0.521, 0.571, 0.502, 0.607, 0.519, 0.662, 0.628, 0.682, 0.756, 0.734, 0.763, 0.641, 0.382],
    },
    'gpt2-large': {
        'params': '774M', 'layers': 36, 'head_dim': 64, 'heads': 20, 'kv_heads': 20,
        'ppl_base': 15.83,
        'results': [
            ('Vanilla K4/V2 w64',    16.11, +1.8, 'vanilla'),
            ('Vanilla K4/V2 w32',    16.17, +2.2, 'vanilla'),
            ('Vanilla K3/V2 w32',    16.39, +3.6, 'vanilla'),
            ('Sink(4)+Heuristic w32', 16.42, +3.8, 'heuristic'),
            ('Sink(8)+Heuristic w32', 16.39, +3.6, 'heuristic'),
            ('Sink(4)+Measured w32',  16.20, +2.4, 'measured'),
            ('Sink(8)+Measured w32',  16.08, +1.6, 'measured'),
            ('Sink(4)+K4/V2 w32',    16.27, +2.8, 'sink_only'),
            ('Sink(8)+K4/V2 w32',    16.20, +2.4, 'sink_only'),
        ],
        'sensitivities': [1.0] + [0.34, 0.32, 0.32, 0.33, 0.32, 0.36, 0.35, 0.38, 0.35, 0.38, 0.37, 0.35, 0.36, 0.35, 0.35, 0.34, 0.34, 0.33, 0.32, 0.32, 0.32, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.24, 0.22, 0.21, 0.19, 0.18],
        'sink_attention': [0.041, 0.072, 0.066, 0.064, 0.062, 0.061, 0.079, 0.124, 0.248, 0.253, 0.356, 0.471, 0.499, 0.478, 0.407, 0.457, 0.552, 0.560, 0.506, 0.591, 0.662, 0.612, 0.545, 0.571, 0.559, 0.683, 0.678, 0.680, 0.713, 0.757, 0.722, 0.712, 0.671, 0.623, 0.431, 0.125],
    },
    'gpt2-xl': {
        'params': '1.5B', 'layers': 48, 'head_dim': 64, 'heads': 25, 'kv_heads': 25,
        'ppl_base': 14.52,
        'results': [
            ('Vanilla K4/V2 w64',    14.70, +1.2, 'vanilla'),
            ('Vanilla K4/V2 w32',    14.75, +1.6, 'vanilla'),
            ('Vanilla K3/V2 w32',    14.84, +2.2, 'vanilla'),
            ('Sink(4)+Heuristic w32', 14.81, +2.0, 'heuristic'),
            ('Sink(8)+Heuristic w32', 14.81, +2.0, 'heuristic'),
            ('Sink(4)+Measured w32',  14.75, +1.6, 'measured'),
            ('Sink(8)+Measured w32',  14.55, +0.2, 'measured'),
            ('Sink(4)+K4/V2 w32',    14.55, +0.2, 'sink_only'),
            ('Sink(8)+K4/V2 w32',    14.64, +0.8, 'sink_only'),
        ],
        'sensitivities': [1.0, 0.03] + [0.02]*6 + [0.03] + [0.02]*10 + [0.01]*28 + [0.005],
        'sink_attention': [0.030, 0.057, 0.060, 0.073, 0.065, 0.093, 0.114, 0.142, 0.195, 0.202, 0.270, 0.316, 0.318, 0.313, 0.367, 0.383, 0.413, 0.444, 0.439, 0.462, 0.482, 0.513, 0.524, 0.546, 0.528, 0.564, 0.566, 0.574, 0.583, 0.605, 0.641, 0.655, 0.660, 0.661, 0.662, 0.753, 0.732, 0.720, 0.714, 0.696, 0.680, 0.722, 0.655, 0.740, 0.612, 0.537, 0.527, 0.103],
    },
    'Qwen/Qwen2.5-1.5B': {
        'params': '1.5B', 'layers': 28, 'head_dim': 128, 'heads': 12, 'kv_heads': 2,
        'ppl_base': 8.36,
        'results': [
            ('Vanilla K4/V2 w64',     1282.0, +15236.1, 'vanilla'),
            ('Vanilla K4/V2 w32',     1458.0, +17341.5, 'vanilla'),
            ('Vanilla K3/V2 w32',     2530.0, +30165.4, 'vanilla'),
            ('Sink(4)+Heuristic w32',    8.88, +6.2, 'heuristic'),
            ('Sink(8)+Heuristic w32',    9.00, +7.7, 'heuristic'),
            ('Sink(4)+Measured w32',     8.90, +6.4, 'measured'),
            ('Sink(8)+Measured w32',     8.83, +5.6, 'measured'),
            ('Sink(4)+K4/V2 w32',     1145.0, +13597.2, 'sink_only'),
            ('Sink(8)+K4/V2 w32',     1186.0, +14087.7, 'sink_only'),
        ],
        'sensitivities': [1.0, 0.59, 0.76, 0.41, 0.36, 0.45, 0.36, 0.36, 0.42, 0.38, 0.38, 0.36, 0.37, 0.34, 0.33, 0.39, 0.35, 0.36, 0.36, 0.35, 0.33, 0.30, 0.29, 0.27, 0.24, 0.20, 0.17, 0.13],
        'sink_attention': None,  # couldn't extract with sdpa
    },
}

OUT = 'results'
os.makedirs(OUT, exist_ok=True)


# ============================================================================
# Figure 1: Cross-model PPL comparison (corrected)
# ============================================================================

def plot_cross_model_comparison():
    fig, ax = plt.subplots(figsize=(14, 7))

    model_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    x = np.arange(len(model_names))
    width = 0.15

    # For each model, get: vanilla K4/V2 w64, vanilla K4/V2 w32, sink(8)+measured, sink(4)+K4/V2
    configs = [
        ('Vanilla K4/V2 w64', 'vanilla', '#aaaaaa'),
        ('Vanilla K4/V2 w32', 'vanilla', '#666666'),
        ('Vanilla K3/V2 w32', 'vanilla', '#333333'),
        ('Sink(8)+Measured w32', 'measured', '#e74c3c'),
        ('Sink(4)+K4/V2 w32', 'sink_only', '#3498db'),
    ]

    for i, (config_name, _, color) in enumerate(configs):
        deltas = []
        for mn in model_names:
            d = models_data[mn]
            match = [r for r in d['results'] if r[0] == config_name]
            if match:
                deltas.append(match[0][2])
            else:
                deltas.append(0)

        bars = ax.bar(x + i * width - 2 * width, deltas, width, label=config_name,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, deltas):
            if abs(val) < 20:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:+.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{mn}\n({models_data[mn]['params']}, {models_data[mn]['layers']}L)"
                        for mn in model_names], fontsize=10)
    ax.set_ylabel('Perplexity Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('TurboQuant-Adaptive: Cross-Model Quality Comparison\n(lower = better, 0% = lossless)',
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(2, color='orange', ls='--', alpha=0.4, lw=1, label='2% threshold')
    ax.axhline(5, color='red', ls='--', alpha=0.3, lw=1, label='5% threshold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.15, axis='y')
    ax.set_ylim(-0.5, max(20, max(r[2] for d in models_data.values() for r in d['results'] if r[2] < 25)) + 1)

    plt.tight_layout()
    plt.savefig(f'{OUT}/cross_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/cross_model_comparison.png")


# ============================================================================
# Figure 2: Sensitivity profiles across models
# ============================================================================

def plot_sensitivity_profiles():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Per-Layer Sensitivity to Quantization Noise\n(Layer 0 is universally most sensitive)',
                 fontsize=14, fontweight='bold')

    model_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'Qwen/Qwen2.5-1.5B']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']

    for idx, (mn, color) in enumerate(zip(model_names, colors)):
        ax = axes[idx // 3][idx % 3]
        d = models_data[mn]
        s = d['sensitivities']
        layers = np.arange(len(s))

        # Color bars by tier
        bar_colors = []
        for v in s:
            if v > 0.5:
                bar_colors.append('#e74c3c')  # critical
            elif v > 0.1:
                bar_colors.append('#e67e22')  # high
            else:
                bar_colors.append('#3498db')  # low

        ax.bar(layers, s, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_title(f"{mn}\n({d['params']}, {d['layers']}L, hd={d['head_dim']})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Sensitivity', fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color='red', ls='--', alpha=0.3, lw=1)
        ax.axhline(0.1, color='orange', ls='--', alpha=0.3, lw=1)
        ax.grid(True, alpha=0.15, axis='y')

    # Legend in empty subplot
    ax = axes[1][2]
    ax.axis('off')
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc='#e74c3c', alpha=0.8, label='Critical (>0.5) → fp16 or K4/V4'),
        plt.Rectangle((0,0), 1, 1, fc='#e67e22', alpha=0.8, label='High (0.1-0.5) → K4/V2'),
        plt.Rectangle((0,0), 1, 1, fc='#3498db', alpha=0.8, label='Low (<0.1) → K3/V2'),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='center', frameon=True,
              title='Bit Allocation Tiers', title_fontsize=13)
    ax.text(0.5, 0.15, 'Layer 0 is ALWAYS most sensitive\nacross all models tested',
            ha='center', va='center', fontsize=11, style='italic', color='#555',
            transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(f'{OUT}/sensitivity_profiles.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/sensitivity_profiles.png")


# ============================================================================
# Figure 3: Attention sink patterns across models
# ============================================================================

def plot_sink_patterns():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Attention Sink Phenomenon: % of Attention on Token 0\n(Deeper layers park more attention on the first token)',
                 fontsize=14, fontweight='bold')

    gpt2_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

    for idx, (mn, color) in enumerate(zip(gpt2_models, colors)):
        ax = axes[idx // 2][idx % 2]
        d = models_data[mn]
        sa = d['sink_attention']
        layers = np.arange(len(sa))

        ax.fill_between(layers, sa, alpha=0.3, color=color)
        ax.plot(layers, sa, color=color, lw=2)

        # Highlight max
        max_idx = np.argmax(sa)
        ax.scatter(max_idx, sa[max_idx], c=color, s=100, zorder=5, edgecolors='black', lw=1.5)
        ax.annotate(f'{sa[max_idx]:.1%}', (max_idx, sa[max_idx]),
                   fontsize=10, fontweight='bold', ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')

        ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
        ax.set_title(f"{mn} ({d['params']}, {d['layers']}L)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Attention on Token 0', fontsize=10)
        ax.set_ylim(0, 0.85)
        ax.grid(True, alpha=0.15)

        # Stats annotation
        avg = np.mean(sa)
        above_50 = sum(1 for x in sa if x > 0.5)
        ax.text(0.02, 0.95, f'avg={avg:.1%}\n{above_50}/{len(sa)} layers >50%',
               transform=ax.transAxes, fontsize=9, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUT}/sink_patterns.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/sink_patterns.png")


# ============================================================================
# Figure 4: Scaling behavior — does improvement increase with model size?
# ============================================================================

def plot_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('How TurboQuant-Adaptive Scales with Model Size',
                 fontsize=14, fontweight='bold', y=1.02)

    gpt2_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    params = [124, 355, 774, 1558]

    # Extract best results per model (CORRECTED — sort by PPL not name)
    vanilla_k4v2_w64 = []
    vanilla_k4v2_w32 = []
    best_ours = []
    sink_only_k4v2 = []

    for mn in gpt2_models:
        d = models_data[mn]
        for name, ppl, delta, cat in d['results']:
            if name == 'Vanilla K4/V2 w64':
                vanilla_k4v2_w64.append(delta)
            if name == 'Vanilla K4/V2 w32':
                vanilla_k4v2_w32.append(delta)
            if name == 'Sink(4)+K4/V2 w32':
                sink_only_k4v2.append(delta)

        ours_results = [r for r in d['results'] if r[3] in ('heuristic', 'measured', 'sink_only')]
        best = min(ours_results, key=lambda x: x[1])
        best_ours.append(best[2])

    # Panel 1: PPL degradation vs model size
    ax = axes[0]
    ax.plot(params, vanilla_k4v2_w64, 'o-', color='#888', label='Vanilla K4/V2 w64', lw=2, markersize=8)
    ax.plot(params, vanilla_k4v2_w32, 's-', color='#555', label='Vanilla K4/V2 w32', lw=2, markersize=8)
    ax.plot(params, best_ours, 'D-', color='#e74c3c', label='Best adaptive (ours)', lw=2.5, markersize=10)
    ax.plot(params, sink_only_k4v2, '^-', color='#3498db', label='Sink(4)+K4/V2 w32', lw=2, markersize=8)

    ax.set_xlabel('Model Parameters (M)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PPL Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality Loss vs Model Size', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(2, color='orange', ls='--', alpha=0.4, label='2% threshold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.set_xscale('log')
    ax.set_xticks(params)
    ax.set_xticklabels(['124M', '355M', '774M', '1.5B'])

    # Panel 2: Improvement gap (vanilla - ours)
    ax = axes[1]
    gap_64 = [v - o for v, o in zip(vanilla_k4v2_w64, best_ours)]
    gap_32 = [v - o for v, o in zip(vanilla_k4v2_w32, best_ours)]

    ax.bar(np.arange(4) - 0.15, gap_64, 0.3, color='#888', alpha=0.8, label='vs K4/V2 w64')
    ax.bar(np.arange(4) + 0.15, gap_32, 0.3, color='#555', alpha=0.8, label='vs K4/V2 w32')

    for i, (g64, g32) in enumerate(zip(gap_64, gap_32)):
        ax.text(i - 0.15, max(g64, 0) + 0.05, f'{g64:+.1f}pp', ha='center', fontsize=8, fontweight='bold')
        ax.text(i + 0.15, max(g32, 0) + 0.05, f'{g32:+.1f}pp', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(['GPT-2\n124M', 'GPT-2 Med\n355M', 'GPT-2 Large\n774M', 'GPT-2 XL\n1.5B'])
    ax.set_ylabel('Improvement (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Improvement Over Vanilla', fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis='y')

    # Panel 3: Layer 0 sensitivity ratio vs model depth
    ax = axes[2]
    l0_ratios = []
    avg_rest = []
    for mn in gpt2_models:
        s = models_data[mn]['sensitivities']
        l0_ratios.append(s[0] / np.mean(s[1:]) if np.mean(s[1:]) > 0 else 0)
        avg_rest.append(np.mean(s[1:]))

    layers_list = [12, 24, 36, 48]
    ax.bar(range(4), l0_ratios, color='#e74c3c', alpha=0.8)
    for i, r in enumerate(l0_ratios):
        ax.text(i, r + 0.5, f'{r:.0f}x', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(4))
    ax.set_xticklabels(['12L', '24L', '36L', '48L'])
    ax.set_ylabel('Layer 0 / Avg(Rest) Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Layer 0 Sensitivity Dominance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/scaling_behavior.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/scaling_behavior.png")


# ============================================================================
# Figure 5: Qwen2.5-1.5B — the GQA failure mode
# ============================================================================

def plot_qwen_analysis():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Qwen2.5-1.5B: GQA Makes Vanilla TurboQuant Catastrophically Fail\n'
                 '(2 KV heads serving 12 query heads — each KV head does 6x more work)',
                 fontsize=13, fontweight='bold', y=1.03)

    d = models_data['Qwen/Qwen2.5-1.5B']

    # Panel 1: PPL comparison (log scale because vanilla is 1000+)
    ax = axes[0]
    configs = []
    for r in d['results']:
        configs.append((r[0], r[1], r[3]))

    names = [c[0] for c in configs]
    ppls = [c[1] for c in configs]
    cats = [c[2] for c in configs]
    colors = {'vanilla': '#e74c3c', 'heuristic': '#2ecc71', 'measured': '#27ae60', 'sink_only': '#e67e22'}

    bars = ax.barh(range(len(names)), ppls, color=[colors.get(c, '#999') for c in cats], alpha=0.85)
    ax.axvline(d['ppl_base'], color='black', ls='--', lw=2, label=f"Baseline={d['ppl_base']:.1f}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Perplexity (log scale)', fontsize=11)
    ax.set_xscale('log')
    ax.set_title('Perplexity by Config', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Panel 2: Sensitivity profile
    ax = axes[1]
    s = d['sensitivities']
    bar_colors = ['#e74c3c' if v > 0.5 else '#e67e22' if v > 0.1 else '#3498db' for v in s]
    ax.barh(range(len(s)), s, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([f'L{i}' for i in range(len(s))], fontsize=7)
    ax.set_xlabel('Sensitivity', fontsize=11)
    ax.set_title('Layer Sensitivity\n(3 critical layers, 25 high)', fontsize=12, fontweight='bold')
    ax.axvline(0.5, color='red', ls='--', alpha=0.4)
    ax.axvline(0.1, color='orange', ls='--', alpha=0.4)
    ax.invert_yaxis()

    # Panel 3: Why it breaks — explanation
    ax = axes[2]
    ax.axis('off')
    explanation = """
WHY VANILLA TURBOQUANT BREAKS ON QWEN

Qwen2.5-1.5B uses Grouped Query Attention (GQA):
  • 12 query heads share 2 KV heads
  • Each KV head serves 6 query heads

This means quantization error in ONE KV head
affects 6x more attention computations.

With standard MHA (GPT-2):
  Error in 1 KV head → affects 1 attention output

With GQA (Qwen, 12:2):
  Error in 1 KV head → affects 6 attention outputs

The 6x amplification factor pushes even K4
quantization past the quality cliff.

OUR FIX: Protect layers 0-2 in fp16 (critical)
         Use K4/V2 for remaining layers (all HIGH)
         Result: PPL 8.83 vs 8.36 baseline (+5.6%)
         vs 1282 for vanilla TurboQuant (+15,236%)
"""
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='#ccc'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/qwen_gqa_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/qwen_gqa_analysis.png")


# ============================================================================
# Figure 6: Honest comparison — corrected "best" metrics
# ============================================================================

def plot_honest_comparison():
    """The FAIR comparison: same base quantizer (K4/V2), only vary the policy."""
    fig, ax = plt.subplots(figsize=(14, 7))

    gpt2_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    x = np.arange(len(gpt2_models))
    width = 0.18

    config_groups = [
        ('Vanilla K4/V2 w32',    '#888888', 'K4/V2 uniform (no sinks, no tiers)'),
        ('Sink(4)+K4/V2 w32',    '#3498db', '+ protect 4 sink tokens in fp16'),
        ('Sink(8)+K4/V2 w32',    '#2980b9', '+ protect 8 sink tokens in fp16'),
        ('Sink(8)+Measured w32',  '#e74c3c', '+ sink protection + adaptive tiers'),
    ]

    for i, (config_name, color, label) in enumerate(config_groups):
        deltas = []
        for mn in gpt2_models:
            match = [r for r in models_data[mn]['results'] if r[0] == config_name]
            deltas.append(match[0][2] if match else 0)

        bars = ax.bar(x + i * width - 1.5 * width, deltas, width,
                      label=label, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{mn}\n({models_data[mn]['params']})" for mn in gpt2_models], fontsize=11)
    ax.set_ylabel('Perplexity Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Fair Comparison: Same Base Quantizer, Different Policies\n'
                 '(all use K4/V2 w32 as base; varies only sink protection + bit tiers)',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(2, color='orange', ls='--', alpha=0.3, lw=1)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.15, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/honest_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {OUT}/honest_comparison.png")


# ============================================================================
# Validation: print corrected summary table
# ============================================================================

def print_corrected_summary():
    print("\n" + "=" * 90)
    print("CORRECTED CROSS-MODEL SUMMARY")
    print("(best vanilla = lowest PPL among vanilla configs, not alphabetical)")
    print("=" * 90)

    print(f"\n  {'Model':25s} {'Base':>7s} {'V K4/V2w64':>11s} {'V K4/V2w32':>11s} "
          f"{'Sink+K4V2':>11s} {'Sink+Adapt':>11s}")
    print("  " + "-" * 82)

    for mn in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'Qwen/Qwen2.5-1.5B']:
        d = models_data[mn]
        row = f"  {mn:25s} {d['ppl_base']:>7.2f}"

        for config_name in ['Vanilla K4/V2 w64', 'Vanilla K4/V2 w32', 'Sink(4)+K4/V2 w32', 'Sink(8)+Measured w32']:
            match = [r for r in d['results'] if r[0] == config_name]
            if match:
                delta = match[0][2]
                if abs(delta) > 100:
                    row += f" {'BROKEN':>11s}"
                else:
                    row += f" {delta:>+10.1f}%"
            else:
                row += f" {'N/A':>11s}"
        print(row)

    # Key comparisons
    print("\n  KEY FINDINGS (GPT-2 family, K4/V2 w32 as baseline comparison):")
    for mn in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        d = models_data[mn]
        vanilla = [r for r in d['results'] if r[0] == 'Vanilla K4/V2 w32'][0][2]
        sink4 = [r for r in d['results'] if r[0] == 'Sink(4)+K4/V2 w32'][0][2]
        adaptive = [r for r in d['results'] if r[0] == 'Sink(8)+Measured w32'][0][2]
        print(f"    {mn:15s}: vanilla={vanilla:+.1f}% → sink={sink4:+.1f}% → adaptive={adaptive:+.1f}%  "
              f"(sink saves {vanilla-sink4:.1f}pp, adaptive saves {vanilla-adaptive:.1f}pp)")

    print("\n  HONEST ASSESSMENT:")
    print("    GPT-2 small:  Sink protection alone gives the biggest win (+2.5% → +1.0%)")
    print("    GPT-2 medium: Our approach is WORSE than vanilla — sensitivity tiers hurt here")
    print("    GPT-2 large:  Measured adaptive helps (+2.2% → +1.6%, 0.6pp improvement)")
    print("    GPT-2 XL:     Strong win (+1.6% → +0.2%, 1.4pp improvement)")
    print("    Qwen 1.5B:    Vanilla TurboQuant completely breaks; our approach makes it work")
    print()
    print("    The improvement is REAL but INCONSISTENT across model sizes.")
    print("    It works best on large models (XL) and GQA models (Qwen).")
    print("    On medium-sized models, the heuristic tier assignment can hurt.")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Generating analysis graphs...")

    plot_cross_model_comparison()
    plot_sensitivity_profiles()
    plot_sink_patterns()
    plot_scaling()
    plot_qwen_analysis()
    plot_honest_comparison()
    print_corrected_summary()

    print(f"\nAll graphs saved to {OUT}/")


if __name__ == "__main__":
    main()
