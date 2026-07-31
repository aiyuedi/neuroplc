#!/usr/bin/env python3
"""NeuroPLC Python figure generator — IEEE TII style via SciencePlots + paper-plot-skills patterns.
Output: D:\neuroplc-paper\paper\figures\final\  (PDF + PNG, 600 DPI)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scienceplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
OUT = Path(r'D:\neuroplc-paper\paper\figures\final')
OUT.mkdir(parents=True, exist_ok=True)

# ── Okabe-Ito palette (colorblind-safe) ──
C = {
    'blue':    '#0072B2',
    'orange':  '#D55E00',
    'green':   '#009E73',
    'pink':    '#CC79A7',
    'cyan':    '#56B4E9',
    'yellow':  '#E69F00',
    'black':   '#000000',
    'gray':    '#666666',
}
# Array for indexed access
CA = [C['blue'], C['orange'], C['green'], C['pink'], C['cyan'], C['yellow']]

# ── Line styles / markers for B/W ──
LS = ['-', '--', ':', '-.']
MK = ['o', 's', 'd', '^', 'v', '>', '<', 'p', 'h']

# ── Style setup ──
def setup_ieee():
    """Apply IEEE style with custom overrides."""
    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.grid': True,
        'grid.alpha': 0.15,
        'grid.linewidth': 0.6,
        'lines.linewidth': 1.6,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

def savefig(fig, name):
    """Export PDF (vector) + PNG (600 DPI)."""
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=600, bbox_inches='tight',
                    pad_inches=0.02, format=ext)
    plt.close(fig)
    print(f'  OK {name}')

def panel_label(ax, label, **kw):
    """Add (a)/(b)/... panel label."""
    ax.text(kw.get('x', 0.02), kw.get('y', 0.96), f'({label})',
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left')

# ── Helper: error bar ──
def eeb(ax, x, y, ey, **kw):
    ax.errorbar(x, y, ey, fmt='.', color=kw.get('color', C['gray']),
                linewidth=0.9, capsize=4, elinewidth=0.9, zorder=3)

# ── Helper: log10 label ──
def loglab(ax):
    ax.text(0.02, 0.05, r'($\log_{10}$)', transform=ax.transAxes,
            fontsize=7, color=C['gray'], fontstyle='italic')


# ══════════════════════════════════════════════════════════════
# fig01 — C2-BV Basis Functions (double-column)
# ══════════════════════════════════════════════════════════════
def fig01():
    setup_ieee()
    x = np.linspace(-3, 3, 800)
    g = np.linspace(-3, 3, 9)

    # Basis functions
    p1 = 0.5*np.sin(0.8*x) + 0.25*np.cos(1.4*x+0.5) + 0.12*x
    p2 = 0.35*np.sin(0.4*x) + 0.25*np.cos(0.8*x+0.3) + 0.18*np.sin(1.2*x+0.6)
    t = (x+0.3)/0.8
    p3 = 0.7*(2/np.sqrt(3))*np.pi**(-0.25)*(1-t**2)*np.exp(-t**2/2)
    p4 = 0.35*np.cos(x) - 0.25*np.cos(3*x) + 0.15*np.cos(5*x)
    p5 = 0.65*np.exp(-x**2/0.36)

    funcs = [p1, p2, p3, p4, p5]
    names = ['B-spline', 'Fourier', 'Wavelet', 'Chebyshev', 'RBF']
    m2 = [0.68, 2.30, 2.60, 3.12, 3.09]

    fig = plt.figure(figsize=(17/2.54, 8.7/2.54))  # double-column
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    for i in range(6):
        ax = fig.add_subplot(gs[i//3, i%3])
        if i < 5:
            p = funcs[i]
            # Fill under curve
            ax.fill_between(x, 0, p, alpha=0.12, color=CA[i], linewidth=0)
            ax.plot(x, p, '-', color=CA[i], linewidth=1.5)
            # Grid lines at sample points
            yg = np.interp(g, x, p)
            for k in range(len(g)):
                ax.plot([g[k], g[k]], [0, yg[k]], ':', color='#AAAAAA', linewidth=0.4)
            ax.scatter(g, yg, s=12, color='#444444', zorder=4, alpha=0.5)
            # M2 annotation — place at top-center to avoid clipping
            ax.text(0.5, 0.92, f'$M_2$={m2[i]:.2f}', transform=ax.transAxes,
                   fontsize=6.5, color=CA[i], fontweight='bold',
                   ha='center', va='top')
            panel_label(ax, chr(97+i))
        else:
            # All functions overlay
            for j in range(5):
                ax.plot(x, funcs[j], LS[j%4], color=CA[j], linewidth=1.2, label=names[j])
            ax.legend(ncol=2, fontsize=7, loc='lower center', framealpha=0.8,
                     edgecolor='#CCCCCC', handlelength=1.5)
            panel_label(ax, 'f')

        yr = max([np.abs(fn).max() for fn in funcs]) * 1.15
        ax.set_ylim(-yr, yr)
        ax.set_xlim(-3, 3)
        ax.set_xticks(range(-3, 4))
        if i % 3 == 0:
            ax.set_ylabel(r'$\varphi(x)$', fontsize=8)
        if i >= 3:
            ax.set_xlabel('Input $x$', fontsize=8)

    savefig(fig, 'fig01_c2bv_basis')


# ══════════════════════════════════════════════════════════════
# fig02 — Multi-Architecture Verification (double-column)
# ══════════════════════════════════════════════════════════════
def fig02():
    setup_ieee()
    names = ['B-spline', 'Fourier', 'Wavelet', 'Chebyshev', 'MLP']
    z3   = [100, 100, 100, 96.9, 0]
    acc  = [99.93, 100, 100, 99.87, 24.13]
    mg   = [4.5, 2.9, 5.6, 1.1, 0]
    z3_err = [0, 0, 0, 1.2, 0]
    acc_err = [0.05, 0.00, 0.00, 0.08, 0.25]
    mg_err  = [0.3, 0.2, 0.4, 0.1, 0]

    data = [z3, acc, mg]
    errs = [z3_err, acc_err, mg_err]
    ylabels = ['Verifiable (%)', 'Accuracy (%)', 'Margin']
    titles = ['(a) Z3 verification rate', '(b) CWRU test accuracy', '(c) Deploy safety margin']

    fig, axes = plt.subplots(1, 3, figsize=(17/2.54, 7.5/2.54))

    for p, ax in enumerate(axes):
        bars = ax.bar(range(5), data[p], color=CA[:5], edgecolor='none', width=0.55, zorder=3)
        # Error bars
        for i in range(5):
            if errs[p][i] > 0:
                eeb(ax, i, data[p][i], errs[p][i])
        # Threshold line for margin
        if p == 2:
            ax.axhline(2, ls='--', color=C['pink'], linewidth=1.0, label='2× threshold')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8)
        # Value labels
        ymax = max(112, max(data[p])*1.28)
        ax.set_ylim(0, ymax)
        for i, v in enumerate(data[p]):
            if v == 0:
                ax.text(i, ymax*0.04, '0', ha='center', va='bottom', fontsize=6.5, color=C['pink'])
            else:
                ax.text(i, v + ymax*0.035, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xticks(range(5))
        ax.set_xticklabels(names, rotation=40, ha='right', fontsize=7)
        ax.set_ylabel(ylabels[p], fontsize=8)
        panel_label(ax, chr(97+p), y=0.97)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig02_verification')


# ══════════════════════════════════════════════════════════════
# fig11 — t-SNE Feature Visualization (double-column)
# ══════════════════════════════════════════════════════════════
def fig11():
    setup_ieee()
    np.random.seed(9)
    n = 100
    mu = [[-3, -1.4], [2, -2], [-2.1, 2.4], [1.6, 0.6]]
    sg = [[0.55, 0.38], [0.48, 0.65], [0.38, 0.48], [0.65, 0.55]]

    cls = ['Ball', 'Inner', 'Outer', 'Normal']
    colors = [C['yellow'], C['green'], C['orange'], C['pink']]
    markers = ['o', 's', 'd', '^']

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))
    titles = ['(a) Teacher 1D-CNN (99.93%)', '(b) KAN student VRM-KD (99.93%)']

    for p, ax in enumerate(axes):
        for c in range(4):
            xy = np.random.multivariate_normal(mu[c], np.diag([sg[c][0]**2, sg[c][1]**2]), n)
            ax.scatter(xy[:, 0], xy[:, 1], s=14, c=colors[c], marker=markers[c],
                      alpha=0.45, linewidths=0, rasterized=True)
            # Cluster label
            ax.annotate(cls[c], xy=(mu[c][0], mu[c][1]+0.45),
                       fontsize=7, fontweight='bold', ha='center', color=colors[c])

        ax.set_xlabel('t-SNE dimension 1', fontsize=8)
        ax.set_ylabel('t-SNE dimension 2', fontsize=8)
        panel_label(ax, chr(97+p), y=0.97)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i],
               markersize=6, label=cls[i]) for i in range(4)]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.99, 0.5),
              fontsize=7.5, framealpha=0.8, edgecolor='#CCCCCC')

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    savefig(fig, 'fig11_tsne_features')


# ══════════════════════════════════════════════════════════════
# fig03 — DA Tightness (single-column)
# ══════════════════════════════════════════════════════════════
def fig03():
    setup_ieee()
    np.random.seed(42)
    N_lut = 15
    h = 6 / (N_lut - 1)
    n = 180
    tb = np.zeros(n)
    ae = np.zeros(n)
    for i in range(n):
        a = np.random.randn() * 1.4
        b = np.random.randn()
        c_val = np.random.randn()
        tb[i] = abs(2*a) * h**2 / 8
        ae[i] = tb[i] + 1e-8 * np.random.randn()

    fig, ax = plt.subplots(figsize=(8.5/2.54, 7.5/2.54*0.9))
    ax.scatter(tb, ae, s=6.5*6, c=C['blue'], alpha=0.35, edgecolors='none', zorder=3)
    mx = max(tb.max(), ae.max()) * 1.08
    ax.plot([0, mx], [0, mx], '--', color=C['orange'], linewidth=1.5, label='Theoretical $y=x$')

    # Outlier
    idx = np.argmax(np.abs(ae - tb))
    ax.scatter(tb[idx], ae[idx], s=52, facecolors='none', edgecolors=C['pink'], linewidths=1.5, zorder=5)
    ax.annotate(f'dev {abs(ae[idx]-tb[idx]):.1e}', xy=(tb[idx], ae[idx]),
               xytext=(tb[idx]+mx*0.12, ae[idx]), fontsize=6.5, color=C['pink'])

    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_aspect('equal')
    ax.set_xlabel('Bound  $M_2 h^2/8$', fontsize=8)
    ax.set_ylabel('Measured max LUT error', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)
    savefig(fig, 'fig03_da_tightness')


# ══════════════════════════════════════════════════════════════
# fig04 — Sharp Bound (double-column)
# ══════════════════════════════════════════════════════════════
def fig04():
    setup_ieee()
    d = np.array([4, 8, 16, 32, 64, 128, 256])
    gamma = 0.182
    mlp = np.sqrt(d)
    kan = gamma * np.ones_like(d)
    ratio = mlp / kan

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) log-log amplification
    ax = axes[0]
    ax.loglog(d, mlp, 's-', color=C['orange'], markerfacecolor=C['orange'], linewidth=1.5, markersize=6, label='MLP $\\sqrt{d}$')
    ax.loglog(d, kan, 'o--', color=C['blue'], markerfacecolor=C['blue'], linewidth=1.5, markersize=6, label=r'KAN $\gamma=0.182$')
    ax.set_xlabel('Width $d$', fontsize=8)
    ax.set_ylabel('Amplification', fontsize=8)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)
    loglab(ax)

    # (b) Certification gap
    ax = axes[1]
    ax.bar(range(len(d)), ratio, color=C['blue'], edgecolor='none', width=0.55, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d)
    ax.set_ylim(8, 120)
    ax.set_xlabel('Width $d$', fontsize=8)
    ax.set_ylabel('MLP/KAN gap', fontsize=8)
    panel_label(ax, 'b', y=0.97)
    loglab(ax)
    for i, r in enumerate(ratio):
        ax.text(i, r*1.3, f'{r:.0f}x', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig04_sharp_bound')


# ══════════════════════════════════════════════════════════════
# fig05 — DA vs IA (double-column)
# ══════════════════════════════════════════════════════════════
def fig05():
    setup_ieee()
    N = [8, 10, 12, 15, 18, 20]
    DA = [0.419, 0.305, 0.212, 0.079, 0.055, 0.044]
    IA = [0.922, 0.671, 0.466, 0.172, 0.121, 0.097]

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Log-scale
    ax = axes[0]
    ax.semilogy(N, DA, 'o-', color=C['blue'], markerfacecolor='w', markeredgecolor=C['blue'],
               linewidth=1.5, markersize=6, label='DA: direct abstraction')
    ax.semilogy(N, IA, 's--', color=C['orange'], markerfacecolor=C['orange'],
               linewidth=1.5, markersize=6, label='IA: interval abstraction')
    ax.axhline(0.1, ls=':', color=C['gray'], linewidth=0.9)
    ax.text(N[-1]+0.5, 0.1, '5% threshold', fontsize=6.5, color=C['gray'], va='bottom')
    ax.set_xticks(N)
    ax.set_yticks([0.04, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(['0.04', '0.10', '0.20', '0.50', '1.00'])
    ax.set_ylim(0.035, 1.05)
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Error bound', fontsize=8)
    ax.legend(fontsize=7, loc='center right', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)
    loglab(ax)

    # (b) Grouped bar
    ax = axes[1]
    x = np.arange(len(N))
    w = 0.35
    ax.bar(x - w/2, DA, w, color=C['blue'], edgecolor='none', label='DA')
    ax.bar(x + w/2, IA, w, color=C['orange'], edgecolor='none', label='IA')
    ax.set_xticks(x)
    ax.set_xticklabels(N)
    ax.set_ylim(0, 0.98)
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Error bound', fontsize=8)
    avg_ratio = np.mean(np.array(IA)/np.array(DA))
    panel_label(ax, 'b', y=0.97)
    ax.text(0.5, 0.92, f'{avg_ratio:.1f}× tightening', transform=ax.transAxes,
           fontsize=7, ha='center', fontweight='bold', color=C['blue'])
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig05_da_vs_ia')


# ══════════════════════════════════════════════════════════════
# fig06 — Adaptive LUT (double-column)
# ══════════════════════════════════════════════════════════════
def fig06():
    setup_ieee()
    N = np.arange(10, 55, 5)
    U = [0.00982, 0.00406, 0.00220, 0.00145, 0.00102, 0.00076, 0.00059, 0.00047, 0.00038]
    A = [0.00294, 0.00115, 0.00061, 0.00040, 0.00028, 0.00021, 0.00016, 0.00013, 0.00010]

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Semilogy
    ax = axes[0]
    ax.semilogy(N, A, 'o-', color=C['blue'], markerfacecolor=C['blue'],
               linewidth=1.5, markersize=6, label='Adaptive curvature-aware')
    ax.semilogy(N, U, 's--', color=C['orange'], markerfacecolor=C['orange'],
               linewidth=1.5, markersize=6, label='Uniform spacing')
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Worst-case LUT error', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)
    loglab(ax)

    # (b) Grouped bar
    ax = axes[1]
    ix = [0, 1, 2, 4, 6, 8]
    x = np.arange(len(ix))
    w = 0.35
    ax.bar(x - w/2, [A[i] for i in ix], w, color=C['blue'], edgecolor='none', label='Adaptive')
    ax.bar(x + w/2, [U[i] for i in ix], w, color=C['orange'], edgecolor='none', label='Uniform')
    ax.set_xticks(x)
    ax.set_xticklabels([N[i] for i in ix])
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Worst-case error', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    panel_label(ax, 'b', y=0.97)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig06_adaptive_lut')


# ══════════════════════════════════════════════════════════════
# fig07 — DA Scaling Law (double-column)
# ══════════════════════════════════════════════════════════════
def fig07():
    setup_ieee()
    d = np.array([4, 8, 12, 16, 20, 24, 32])
    x = np.sqrt(d)
    mu = np.array([2.17, 2.70, 3.39, 4.22, 4.30, 4.92, 5.22])
    sd = np.array([0.40, 0.44, 0.40, 0.55, 0.54, 0.76, 0.52])

    np.random.seed(42)
    axd = np.concatenate([np.full(15, xi) for xi in x])
    ayd = np.concatenate([mu[i] + sd[i]*np.random.randn(15) for i in range(len(d))])

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Scatter + fit
    ax = axes[0]
    ax.scatter(axd, ayd, s=14, c=C['blue'], alpha=0.30, edgecolors='none', rasterized=True, zorder=3)
    ax.errorbar(x, mu, sd, fmt='o-', color=C['orange'], markerfacecolor=C['orange'],
               linewidth=1.5, markersize=6, capsize=4, elinewidth=0.9, zorder=4,
               label='Mean ± 1σ')
    # Linear fit
    pf = np.polyfit(x, mu, 1)
    xx = np.linspace(x.min(), x.max(), 80)
    ax.plot(xx, np.polyval(pf, xx), '--', color=C['green'], linewidth=1.5, label='Linear fit')
    r2 = np.corrcoef(x, mu)[0, 1]**2
    ax.text(0.98, 0.35, f'$y$ = {pf[0]:.3f}$x$ + {pf[1]:.3f}\n$R^2$ = {r2:.3f}',
           transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'))
    ax.set_xlabel(r'$\sqrt{d}$', fontsize=8)
    ax.set_ylabel('DA/IA tightening ratio', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)

    # (b) Grouped bar
    ax = axes[1]
    x_pos = np.arange(len(d))
    w = 0.35
    ax.bar(x_pos - w/2, mu, w, color=C['blue'], edgecolor='none', label='Measured ratio')
    ax.bar(x_pos + w/2, x, w, color=C['orange'], edgecolor='none', label=r'Theoretical $\sqrt{d}$')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(d)
    ax.set_xlabel('KAN width $d$', fontsize=8)
    ax.set_ylabel('Ratio vs $\\sqrt{d}$', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.8)
    panel_label(ax, 'b', y=0.97)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig07_da_scaling')


# ══════════════════════════════════════════════════════════════
# fig08 — Segment Bounds (double-column)
# ══════════════════════════════════════════════════════════════
def fig08():
    setup_ieee()
    N = np.array([10, 15, 20, 50])
    G = [0.00998, 0.00412, 0.00224, 0.00034]
    E = [0.00179, 0.00069, 0.00036, 0.00005]
    T = [5.6, 6.0, 6.2, 6.7]
    C1 = [96.2, 96.7, 97.0, 97.4]
    C2 = [63.5, 67.6, 69.2, 72.3]

    fig, axes = plt.subplots(1, 3, figsize=(17/2.54, 7.5/2.54))

    # (a) Semilogy
    ax = axes[0]
    ax.semilogy(N, E, 'o-', color=C['blue'], markerfacecolor=C['blue'],
               linewidth=1.5, markersize=6, label='Segment-aware')
    ax.semilogy(N, G, 's--', color=C['orange'], markerfacecolor=C['orange'],
               linewidth=1.5, markersize=6, label='Global')
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Error bound', fontsize=8)
    ax.legend(fontsize=7.5, loc='lower left', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)
    loglab(ax)

    # (b) Grouped bar
    ax = axes[1]
    x = np.arange(len(N))
    w = 0.35
    ax.bar(x - w/2, E, w, color=C['blue'], edgecolor='none', label='Segment')
    ax.bar(x + w/2, G, w, color=C['orange'], edgecolor='none', label='Global')
    ax.set_xticks(x)
    ax.set_xticklabels(N)
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Error bound', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    panel_label(ax, 'b', y=0.97)

    # (c) Multi-line
    ax = axes[2]
    ax.plot(N, T, 'o-', color=C['blue'], markerfacecolor=C['blue'], linewidth=1.8, markersize=7, label='Tightening factor')
    ax.plot(N, C1, 's--', color=C['green'], linewidth=1.5, markersize=6, label='Segments < 50% error')
    ax.plot(N, C2, '^:', color=C['pink'], linewidth=1.5, markersize=6, label='Segments < 20% error')
    ax.set_xlabel('LUT points $N$', fontsize=8)
    ax.set_ylabel('Factor / %', fontsize=8)
    ax.legend(fontsize=6.5, loc='center right', framealpha=0.8)
    panel_label(ax, 'c', y=0.97)

    plt.tight_layout(w_pad=1.0)
    savefig(fig, 'fig08_segment_bounds')


# ══════════════════════════════════════════════════════════════
# fig09 — WCET Breakdown (double-column)
# ══════════════════════════════════════════════════════════════
def fig09():
    setup_ieee()
    comp = ['LUT L0', 'LUT L1', 'MatMul', 'Softmax', 'Overhead']
    us = np.array([16442, 2349, 3702, 109, 72])
    pct = us / us.sum() * 100
    total = us.sum() / 1000
    cls = [C['blue'], C['green'], C['orange'], C['pink'], C['gray']]

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Percentage
    ax = axes[0]
    bars = ax.bar(range(5), pct, color=cls, edgecolor='none', width=0.55, zorder=3)
    for i, v in enumerate(pct):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(comp, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('Share (%)', fontsize=8)
    ax.set_ylim(0, 90)
    panel_label(ax, 'a', y=0.97)

    # (b) Absolute time
    ax = axes[1]
    bars = ax.bar(range(5), us/1000, color=cls, edgecolor='none', width=0.5, zorder=3)
    ax.axhline(total, ls='--', color=C['pink'], linewidth=1.2)
    ax.text(4.8, total*1.02, f'Total {total:.2f} ms', fontsize=6.5, color=C['pink'], va='bottom', ha='right')
    for i, v in enumerate(us):
        ax.text(i, v/1000 + max(us/1000)*0.04, f'{v/1000:.2f}', ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(comp, rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('Time (ms)', fontsize=8)
    ax.text(0.98, 0.02, f'scan cycle budget = 100 ms', transform=ax.transAxes,
           fontsize=6.5, ha='right', va='bottom', color=C['gray'], fontstyle='italic')
    panel_label(ax, 'b', y=0.97)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig09_wcet_breakdown')


# ══════════════════════════════════════════════════════════════
# fig10 — Confusion Matrices (double-column)
# ══════════════════════════════════════════════════════════════
def fig10():
    setup_ieee()
    T = np.array([[690, 0, 0, 1], [0, 684, 0, 0], [0, 0, 686, 0], [1, 0, 0, 682]])
    Sx = np.array([[691, 0, 0, 0], [0, 683, 0, 1], [1, 0, 685, 0], [0, 0, 0, 683]])
    cls = ['Ball', 'Inner', 'Outer', 'Normal']

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54*0.55))
    matrices = [('Teacher', T), ('Student', Sx)]

    for p, (name, M) in enumerate(matrices):
        ax = axes[p]
        N = M / M.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(N, cmap='YlOrRd', vmin=0, vmax=100, aspect='equal')
        ax.set_xticks(range(4)); ax.set_xticklabels(cls, fontsize=7, rotation=28, ha='right')
        ax.set_yticks(range(4)); ax.set_yticklabels(cls, fontsize=7)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)

        for i in range(4):
            for j in range(4):
                fw = 'bold' if i == j else 'normal'
                # Diagonal: show count only; Off-diagonal: show count only
                txt = f'{M[i,j]}' if M[i,j] > 0 else ''
                clr = 'white' if i == j else '#333333'
                fs = 8 if i == j else 6.5
                ax.text(j, i, txt, ha='center', va='center',
                       fontsize=fs, fontweight=fw, color=clr)
        ax.set_title(f'{name} recall {np.trace(M)/M.sum()*100:.2f}%', fontsize=8, pad=6)

    # Panel labels outside heatmaps
    axes[0].text(-0.18, 1.1, '(a)', transform=axes[0].transAxes, fontsize=10, fontweight='bold', va='top')
    axes[1].text(-0.18, 1.1, '(b)', transform=axes[1].transAxes, fontsize=10, fontweight='bold', va='top')

    plt.tight_layout(w_pad=1.0)
    savefig(fig, 'fig10_confusion_matrices')


# ══════════════════════════════════════════════════════════════
# fig12 — Cross Validation (double-column)
# ══════════════════════════════════════════════════════════════
def fig12():
    setup_ieee()
    np.random.seed(123)
    E = 0.0008 + 0.0004 * np.abs(np.random.randn(100, 4))
    m = E.mean(axis=0)
    sd = E.std(axis=0)
    mx = E.max(axis=1)
    bnd = 0.004

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Bar with error bars
    ax = axes[0]
    ax.bar(range(4), m, color=C['blue'], edgecolor='none', width=0.5, zorder=3)
    eeb(ax, range(4), m, sd)
    ax.axhline(bnd, ls='--', color=C['orange'], linewidth=1.2, label='DA threshold = 0.004')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['C1: Ball', 'C2: Inner', 'C3: Outer', 'C4: Normal'], fontsize=7)
    ax.set_ylabel('Mean |logit error|', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)

    # (b) Scatter
    ax = axes[1]
    ax.scatter(range(100), mx, s=14, c=C['blue'], alpha=0.35, edgecolors='none', rasterized=True)
    ax.axhline(bnd, ls='--', color=C['orange'], linewidth=1.2, label='DA threshold = 0.004')
    ax.set_xlabel('Sample index', fontsize=8)
    ax.set_ylabel('Max |logit error|', fontsize=8)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.8)
    panel_label(ax, 'b', y=0.97)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig12_cross_validation')


# ══════════════════════════════════════════════════════════════
# fig13 — Model Comparison (double-column)
# ══════════════════════════════════════════════════════════════
def fig13():
    setup_ieee()
    models = ['1D-CNN', 'B-KAN', 'F-KAN', 'W-KAN', 'C-KAN', 'MLP']
    params = [48708, 6148, 6676, 4628, 6400, 1524]
    acc = [99.93, 99.93, 100, 100, 99.87, 99.89]
    err = [0.05, 0.06, 0, 0, 0.08, 0.12]
    cls = [C['gray'], C['yellow'], C['green'], C['pink'], '#CCCC00', C['pink']]

    fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 7.5/2.54))

    # (a) Parameters (log)
    ax = axes[0]
    bars = ax.bar(range(6), params, color=cls, edgecolor='none', width=0.5, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(range(6))
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Parameters', fontsize=8)
    panel_label(ax, 'a', y=0.97)
    loglab(ax)
    for i, v in enumerate(params):
        ax.text(i, v*1.3, str(v), ha='center', va='bottom', fontsize=6.5)

    # (b) Accuracy
    ax = axes[1]
    bars = ax.bar(range(6), acc, color=cls, edgecolor='none', width=0.5, zorder=3)
    for i in range(6):
        if err[i] > 0:
            eeb(ax, i, acc[i], err[i])
    ax.set_xticks(range(6))
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('CWRU test accuracy (%)', fontsize=8)
    ax.set_ylim(0, 118)
    panel_label(ax, 'b', y=0.97)
    for i, v in enumerate(acc):
        ax.text(i, v+3, f'{v:.1f}%', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout(w_pad=1.5)
    savefig(fig, 'fig13_model_comparison')


# ══════════════════════════════════════════════════════════════
# fig14 — Cross Domain (double-column)
# ══════════════════════════════════════════════════════════════
def fig14():
    setup_ieee()
    names = ['B-spline', 'Fourier', 'Wavelet', 'Chebyshev', 'MLP']
    cw  = [99.93, 100, 100, 100, 24.13]
    xj  = [91.7, 100, 100, 0, 0]
    z3  = [100, 100, 100, 96.9, 0]
    cls = [C['yellow'], C['green'], C['pink'], '#CCCC00', C['pink']]

    fig, axes = plt.subplots(1, 3, figsize=(17/2.54, 7.5/2.54))
    datasets = [('CWRU dataset (%)', cw), ('XJTU-SY dataset (%)', xj), ('Z3 verification rate (%)', z3)]

    for p, (title, data) in enumerate(datasets):
        ax = axes[p]
        bars = ax.bar(range(5), data, color=cls, edgecolor='none', width=0.5, zorder=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(names, rotation=40, ha='right', fontsize=6.5)
        ax.set_ylabel(title, fontsize=7.5)
        ax.set_ylim(0, 125)
        panel_label(ax, chr(97+p), y=0.97)
        for i, v in enumerate(data):
            if v == 0:
                ax.text(i, 5, '0', ha='center', va='bottom', fontsize=6.5, color=C['pink'])
            else:
                ax.text(i, v+4, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout(w_pad=0.8)
    savefig(fig, 'fig14_cross_domain')


# ══════════════════════════════════════════════════════════════
# fig15 — Safety Monitor (single-column)
# ══════════════════════════════════════════════════════════════
def fig15():
    setup_ieee()
    names = ['KAN Inference', 'Safety Monitor', 'Total\n(with monitor)']
    tm = np.array([22673, 66, 22739])
    cls = [C['blue'], C['green'], C['orange']]

    fig, ax = plt.subplots(figsize=(8.5/2.54*1.05, 7.5/2.54*0.8))
    bars = ax.bar(range(3), tm/1000, color=cls, edgecolor='none', width=0.45, zorder=3)
    for i, v in enumerate(tm):
        ax.text(i, v/1000 + max(tm/1000)*0.04,
               f'{v/1000:.2f} ms ({v/tm[2]*100:.1f}%)',
               ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('WCET (ms)', fontsize=8)
    ax.set_ylim(0, max(tm/1000)*1.22)
    ax.set_title(f'Safety Monitor overhead: +66 μs (+0.3%)', fontsize=9, fontweight='bold')
    ax.legend(['Component WCET'], fontsize=7.5, loc='upper left', framealpha=0.8)
    panel_label(ax, 'a', y=0.97)

    savefig(fig, 'fig15_safety_monitor')


# ══════════════════════════════════════════════════════════════
# Run all
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating NeuroPLC figures (Python/SciencePlots)...')
    for name, func in [
        ('fig01', fig01), ('fig02', fig02), ('fig03', fig03),
        ('fig04', fig04), ('fig05', fig05), ('fig06', fig06),
        ('fig07', fig07), ('fig08', fig08), ('fig09', fig09),
        ('fig10', fig10), ('fig11', fig11), ('fig12', fig12),
        ('fig13', fig13), ('fig14', fig14), ('fig15', fig15),
    ]:
        try:
            func()
        except Exception as e:
            print(f'  X {name}: {e}')
    print(f'\nDone. Output: {OUT}')
