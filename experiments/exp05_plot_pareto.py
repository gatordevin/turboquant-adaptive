"""
Clean TurboQuant Pareto frontier plots from benchmark results.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ── Data from benchmark run ──
results = [
    # (name, key_bits, value_bits, window, compression, ppl, throughput)
    ("Baseline fp16",   16, 16, 9999, 1.00, 23.62, 168.7),
    ("K1/V1 w16",        1,  1,   16, 9.35, 149.00, 14.0),
    ("K1/V1 w32",        1,  1,   32, 7.37, 138.38, 17.3),
    ("K1/V1 w64",        1,  1,   64, 5.17,  98.12, 24.7),
    ("K1/V1 w128",       1,  1,  128, 3.24,  55.91, 105.7),
    ("K2/V1 w16",        2,  1,   16, 7.29,  50.59, 13.9),
    ("K2/V1 w32",        2,  1,   32, 6.06,  47.62, 17.2),
    ("K2/V1 w64",        2,  1,   64, 4.53,  39.56, 28.4),
    ("K2/V1 w128",       2,  1,  128, 3.01,  32.41, 106.4),
    ("K2/V2 w16",        2,  2,   16, 5.97,  43.28, 13.9),
    ("K2/V2 w32",        2,  2,   32, 5.15,  41.47, 17.2),
    ("K2/V2 w64",        2,  2,   64, 4.03,  37.44, 28.5),
    ("K2/V2 w128",       2,  2,  128, 2.81,  29.34, 105.6),
    ("K3/V1 w16",        3,  1,   16, 5.97,  38.78, 13.6),
    ("K3/V1 w32",        3,  1,   32, 5.15,  37.31, 17.3),
    ("K3/V1 w64",        3,  1,   64, 4.03,  32.84, 28.7),
    ("K3/V1 w128",       3,  1,  128, 2.81,  28.05, 108.2),
    ("K3/V2 w16",        3,  2,   16, 5.06,  27.67, 14.0),
    ("K3/V2 w32",        3,  2,   32, 4.47,  27.61, 17.3),
    ("K3/V2 w64",        3,  2,   64, 3.63,  26.97, 28.6),
    ("K3/V2 w128",       3,  2,  128, 2.64,  25.39, 106.2),
    ("K3/V3 w16",        3,  3,   16, 4.39,  27.08, 13.5),
    ("K3/V3 w32",        3,  3,   32, 3.95,  27.19, 17.1),
    ("K3/V3 w64",        3,  3,   64, 3.30,  26.77, 28.0),
    ("K3/V3 w128",       3,  3,  128, 2.49,  24.91, 104.0),
    ("K4/V1 w16",        4,  1,   16, 5.06,  31.59, 13.9),
    ("K4/V1 w32",        4,  1,   32, 4.47,  30.62, 17.4),
    ("K4/V1 w64",        4,  1,   64, 3.63,  29.05, 28.7),
    ("K4/V1 w128",       4,  1,  128, 2.64,  26.97, 107.4),
    ("K4/V2 w16",        4,  2,   16, 4.39,  24.61, 13.9),
    ("K4/V2 w32",        4,  2,   32, 3.95,  24.56, 16.6),
    ("K4/V2 w64",        4,  2,   64, 3.30,  24.14, 28.7),
    ("K4/V2 w128",       4,  2,  128, 2.49,  24.09, 108.4),
    ("K4/V3 w16",        4,  3,   16, 3.87,  24.47, 14.0),
    ("K4/V3 w32",        4,  3,   32, 3.54,  24.52, 17.2),
    ("K4/V3 w64",        4,  3,   64, 3.03,  24.28, 28.6),
    ("K4/V3 w128",       4,  3,  128, 2.35,  24.38, 107.0),
    ("K4/V4 w16",        4,  4,   16, 3.47,  24.19, 13.9),
    ("K4/V4 w32",        4,  4,   32, 3.21,  24.14, 17.4),
    ("K4/V4 w64",        4,  4,   64, 2.80,  24.05, 28.7),
    ("K4/V4 w128",       4,  4,  128, 2.23,  24.09, 107.3),
]

ppl_base = 23.62
tps_base = 168.7

# ── Styling ──
kb_colors = {1: '#e74c3c', 2: '#e67e22', 3: '#27ae60', 4: '#2980b9', 16: '#1a1a2e'}
win_markers = {16: 'v', 32: 's', 64: 'D', 128: '^', 9999: '*'}
win_sizes = {16: 55, 32: 55, 64: 55, 128: 60, 9999: 200}

# ── Find Pareto front (compression vs ppl, lower ppl better) ──
sorted_by_comp = sorted(results, key=lambda r: r[4])  # by compression
pareto_ppl = []
best_ppl = float('inf')
for r in sorted_by_comp:
    if r[5] <= best_ppl:
        best_ppl = r[5]
        pareto_ppl.append(r)

pareto_names = set(r[0] for r in pareto_ppl)

# ============================================================================
fig = plt.figure(figsize=(22, 14))

# Use gridspec for layout: 2 rows
# Top row: 3 panels
# Bottom row: 1 wide panel (zoomed sweet spot)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.85], hspace=0.35, wspace=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: Compression vs Perplexity (full view)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

for r in results:
    name, kb, vb, win, comp, ppl, tps = r
    c = kb_colors[kb]
    m = win_markers[win]
    sz = win_sizes[win]
    is_pareto = name in pareto_names
    is_base = kb == 16

    ax1.scatter(comp, ppl, c=c, marker=m, s=sz * (2.5 if is_base else 1),
                edgecolors='black' if is_pareto else 'none',
                linewidths=2 if is_pareto else 0, zorder=4 if is_pareto else 3, alpha=0.85)

# Pareto line
px = [r[4] for r in pareto_ppl]
py = [r[5] for r in pareto_ppl]
ax1.plot(px, py, 'k--', alpha=0.5, lw=2, label='Pareto frontier', zorder=2)

ax1.axhline(ppl_base, color='#555', ls=':', alpha=0.6, lw=1)
ax1.axhline(ppl_base * 1.05, color='#e67e22', ls='--', alpha=0.5, lw=1.5, label='5% degradation')
ax1.fill_between([0, 12], ppl_base, ppl_base * 1.05, alpha=0.08, color='green', label='< 5% PPL loss')

ax1.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Perplexity (↓ better)', fontsize=12, fontweight='bold')
ax1.set_title('Quality vs Compression', fontsize=14, fontweight='bold')
ax1.set_ylim(20, 160)
ax1.set_xlim(0.5, 10)
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.2)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: Compression vs Throughput
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

for r in results:
    name, kb, vb, win, comp, ppl, tps = r
    c = kb_colors[kb]
    m = win_markers[win]
    sz = win_sizes[win]
    is_base = kb == 16

    ax2.scatter(comp, tps, c=c, marker=m, s=sz * (2.5 if is_base else 1),
                edgecolors='black' if is_base else 'none',
                linewidths=2 if is_base else 0, zorder=4 if is_base else 3, alpha=0.85)

ax2.axhline(tps_base, color='#555', ls=':', alpha=0.6, lw=1, label=f'Baseline = {tps_base:.0f} tok/s')

# Annotate the throughput clusters
ax2.annotate('win=128\n(mostly fp16)', xy=(2.5, 107), fontsize=8, color='#555',
             ha='center', style='italic')
ax2.annotate('win=16-64\n(heavy compression\nbut dequant overhead)', xy=(5, 22), fontsize=8,
             color='#555', ha='center', style='italic')

ax2.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Throughput (tok/s, ↑ better)', fontsize=12, fontweight='bold')
ax2.set_title('Speed vs Compression', fontsize=14, fontweight='bold')
ax2.set_xlim(0.5, 10)
ax2.legend(fontsize=8, loc='center right')
ax2.grid(True, alpha=0.2)

# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: PPL vs Throughput colored by compression
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

compressions = [r[4] for r in results]
norm = plt.Normalize(1, max(compressions))
cmap = plt.cm.plasma

for r in results:
    name, kb, vb, win, comp, ppl, tps = r
    c = cmap(norm(comp))
    m = win_markers[win]
    sz = win_sizes[win]
    is_base = kb == 16

    ax3.scatter(ppl, tps, c=[c], marker=m, s=sz * (2.5 if is_base else 1),
                edgecolors='black' if is_base else 'none',
                linewidths=2 if is_base else 0, zorder=4 if is_base else 3, alpha=0.85)

    if is_base:
        ax3.annotate('Baseline\nfp16', (ppl, tps), fontsize=9, fontweight='bold',
                     ha='right', va='bottom', xytext=(-12, 5), textcoords='offset points')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax3, shrink=0.8, pad=0.02)
cbar.set_label('Compression (×)', fontsize=10)

ax3.axvline(ppl_base * 1.05, color='#e67e22', ls='--', alpha=0.5, lw=1.5, label='5% PPL')
ax3.axhline(tps_base, color='#555', ls=':', alpha=0.4, lw=1)

# Ideal region
ax3.annotate('IDEAL\nREGION', xy=(23, 170), fontsize=10, color='green',
             fontweight='bold', alpha=0.4, ha='center')

ax3.set_xlabel('Perplexity (← better)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Throughput (tok/s, ↑ better)', fontsize=12, fontweight='bold')
ax3.set_title('Speed vs Quality', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)

# ─────────────────────────────────────────────────────────────────────────────
# Bottom panel: Zoomed sweet-spot region
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :])

# Only show configs with PPL < 35 (the interesting region)
sweet_results = [r for r in results if r[5] < 35]

for r in sweet_results:
    name, kb, vb, win, comp, ppl, tps = r
    c = kb_colors[kb]
    m = win_markers[win]
    sz = win_sizes[win] * 2
    is_pareto = name in pareto_names
    is_base = kb == 16

    ax4.scatter(comp, ppl, c=c, marker=m, s=sz * (2 if is_base else 1),
                edgecolors='black' if is_pareto else 'none',
                linewidths=2.5 if is_pareto else 0, zorder=4 if is_pareto else 3, alpha=0.85)

    # Label interesting points
    if is_base or (is_pareto and comp > 2.5) or name in ['K4/V2 w64', 'K4/V2 w128', 'K3/V2 w128',
                                                           'K4/V4 w64', 'K3/V3 w128']:
        va = 'bottom' if ppl < 26 else 'top'
        offset = (8, 6) if va == 'bottom' else (8, -10)
        ax4.annotate(name, (comp, ppl), fontsize=8.5, fontweight='bold' if is_pareto else 'normal',
                     ha='left', va=va, xytext=offset, textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8) if not is_base else None,
                     color=c)

# Pareto line (only sweet spot portion)
sweet_pareto = [r for r in pareto_ppl if r[5] < 35]
px = [r[4] for r in sweet_pareto]
py = [r[5] for r in sweet_pareto]
ax4.plot(px, py, 'k--', alpha=0.5, lw=2, label='Pareto frontier')

ax4.axhline(ppl_base, color='#555', ls=':', alpha=0.6, lw=1, label=f'Baseline PPL = {ppl_base}')
ax4.axhline(ppl_base * 1.02, color='#27ae60', ls='--', alpha=0.4, lw=1.5, label='2% degradation')
ax4.axhline(ppl_base * 1.05, color='#e67e22', ls='--', alpha=0.5, lw=1.5, label='5% degradation')
ax4.axhline(ppl_base * 1.10, color='#e74c3c', ls='--', alpha=0.4, lw=1.5, label='10% degradation')

ax4.fill_between([0, 10], ppl_base, ppl_base * 1.02, alpha=0.08, color='#27ae60')
ax4.fill_between([0, 10], ppl_base * 1.02, ppl_base * 1.05, alpha=0.06, color='#e67e22')

ax4.set_xlabel('Compression Ratio (×)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Perplexity (↓ better)', fontsize=13, fontweight='bold')
ax4.set_title('SWEET SPOT ZONE — Quality vs Compression (zoomed)', fontsize=14, fontweight='bold')
ax4.set_xlim(0.8, 5.5)
ax4.set_ylim(23, 35)
ax4.legend(fontsize=9, loc='upper left', ncol=2)
ax4.grid(True, alpha=0.2)

# ─── Shared legend ──
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[1], markersize=10, label='Keys = 1 bit'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[2], markersize=10, label='Keys = 2 bit'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[3], markersize=10, label='Keys = 3 bit'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=kb_colors[4], markersize=10, label='Keys = 4 bit'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor=kb_colors[16], markersize=14, label='Baseline fp16'),
    Line2D([], [], color='none', label='   '),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=9, label='Residual win = 16'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=9, label='Residual win = 32'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=9, label='Residual win = 64'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=9, label='Residual win = 128'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=10, fontsize=9,
           frameon=True, fancybox=True, edgecolor='#ccc',
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('TurboQuant KV-Cache Compression — GPT-2 Pareto Frontiers',
             fontsize=18, fontweight='bold', y=1.01)

plt.savefig('/home/farmspace/aitest/turboquant_pareto.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Saved: turboquant_pareto.png")
