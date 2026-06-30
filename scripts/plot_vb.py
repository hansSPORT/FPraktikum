"""
V(B) Messungen — Bleifilm Bc(T) Phasendiagramm
Pb-Film (100 nm), He-Kryostat, 4-Punkt-Messung
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import glob, os, re

DATA_DIR = "/home/user/Uni-Projekte/privat/Fpraktikum/Praktikumsdaten Sortiert/V-B"
FIG_DIR  = "/home/user/FPraktikum/figures"
os.makedirs(FIG_DIR, exist_ok=True)

B_RATE_mT_per_s = 0.05 * 0.10524 * 1000  # 5.262 mT/s
V_SC_MAX  = 2e-3   # V
V_N_MIN   = 3e-3   # V
V_RANGE_MIN = 1e-3 # V

def load_vb_file(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 5:
                try:
                    data.append((float(p[0]), float(p[2]), float(p[4])))
                except ValueError:
                    continue
    return np.array(data)

def extract_temperature(filename):
    m = re.search(r'data_[+]?([0-9]+\.[0-9]+)', os.path.basename(filename))
    return float(m.group(1)) if m else None

def find_bc_50percent(B_mT, V):
    V_start = np.median(V[:8])
    V_end   = np.median(V[-8:])
    if V_start > V_N_MIN and V_end > V_N_MIN:
        return None, 'already_normal'
    if V_start < V_SC_MAX and V_end < V_SC_MAX:
        return None, 'no_transition'
    if (V_end - V_start) < V_RANGE_MIN:
        return None, 'no_transition'
    V_thresh = 0.5 * (V_start + V_end)
    idx = np.where(V > V_thresh)[0]
    if len(idx) == 0:
        return None, 'no_transition'
    i = idx[0]
    if i == 0:
        return B_mT[0], 'ok'
    frac = (V_thresh - V[i-1]) / (V[i] - V[i-1])
    return B_mT[i-1] + frac * (B_mT[i] - B_mT[i-1]), 'ok'

# --- Laden ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.dat")))
all_data = []
temps_fit, Bc_fit = [], []

for f in files:
    T = extract_temperature(f)
    if T is None:
        continue
    data = load_vb_file(f)
    if len(data) < 5:
        continue
    t_arr = data[:, 0]
    V_arr = data[:, 1]
    B_mT  = B_RATE_mT_per_s * t_arr
    Bc, status = find_bc_50percent(B_mT, V_arr)
    all_data.append((T, B_mT, V_arr, Bc, status))
    if status == 'ok':
        temps_fit.append(T)
        Bc_fit.append(Bc)

temps_fit = np.array(temps_fit)
Bc_fit    = np.array(Bc_fit)

# --- Bc(T) Fit ---
def bc_parabola(T, Bc0, Tc):
    return Bc0 * (1 - (T / Tc)**2)

popt, pcov = curve_fit(bc_parabola, temps_fit, Bc_fit, p0=[1200.0, 6.75],
                       bounds=([100, 5.0], [5000, 9.0]))
perr = np.sqrt(np.diag(pcov))
Bc0_fit, Tc_fit = popt
Bc0_err, Tc_err = perr
print(f"Bc(0) = {Bc0_fit:.1f} ± {Bc0_err:.1f} mT")
print(f"Tc    = {Tc_fit:.4f} ± {Tc_err:.4f} K")

# ======================================================
# FIGURE 1: V(B) Kurven
# Colormap: 'turbo' — kein Weiß in der Mitte
# ======================================================
fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
cmap = matplotlib.colormaps['turbo']
T_all = [d[0] for d in all_data]
T_min_c, T_max_c = min(T_all), max(T_all)

for T, B_mT, V_arr, Bc, status in all_data:
    color = cmap((T - T_min_c) / (T_max_c - T_min_c))
    if status == 'ok':
        ax1.plot(B_mT, V_arr * 1e3, color=color, lw=1.5, alpha=0.9, zorder=3)
        if Bc is not None:
            Bc_V = np.interp(Bc, B_mT, V_arr)
            ax1.plot(Bc, Bc_V * 1e3, 'o', color=color, ms=6,
                     markeredgecolor='k', markeredgewidth=0.7, zorder=5)
    elif status == 'already_normal':
        ax1.plot(B_mT, V_arr * 1e3, color=color, lw=1.0, ls=':', alpha=0.4, zorder=2)
    elif status == 'no_transition':
        ax1.plot(B_mT, V_arr * 1e3, color=color, lw=1.0, ls='--', alpha=0.55, zorder=2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_min_c, T_max_c))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1)
cbar.set_label('$T$ (K)', fontsize=12)

legend_els = [
    Line2D([0],[0], color='dimgray', lw=1.5, label='SC→N Übergang'),
    Line2D([0],[0], color='dimgray', lw=1.0, ls='--', alpha=0.7,
           label=r'kein Übergang ($B_c > B_\mathrm{max}$)'),
    Line2D([0],[0], color='dimgray', lw=1.0, ls=':', alpha=0.6,
           label='bereits normal bei $B=0$'),
    Line2D([0],[0], marker='o', color='dimgray', ms=6, lw=0,
           markeredgecolor='k', label='$B_c$ (50%-Schwelle)'),
]
ax1.legend(handles=legend_els, fontsize=9, loc='upper left')
ax1.set_xlabel(r'$\mu_0 H$ (mT)', fontsize=13)
ax1.set_ylabel('$U$ (mV)', fontsize=13)
ax1.set_title('Spannungstransition Pb-Film: $U(B)$ bei verschiedenen Temperaturen', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "fig_vb_curves.pdf"), dpi=150)
fig1.savefig(os.path.join(FIG_DIR, "fig_vb_curves.png"), dpi=150)
print("Gespeichert: fig_vb_curves")

# ======================================================
# FIGURE 2: Bc(T) Phasendiagramm
# Fixes:
# - Fit nur im Datenbereich geplottet (kein wilder Extrapolationsstart)
# - T=6.901K (physikalisch anomal: > Tc) als offener Marker
# - Bulk-Linie besser sichtbar annotiert
# ======================================================
fig2, ax2 = plt.subplots(figsize=(7, 5))

# Mess-Punkte aufteilen: normale vs. anomale (T > Tc_fit)
mask_normal  = temps_fit <= Tc_fit
mask_anomal  = temps_fit >  Tc_fit

ax2.plot(temps_fit[mask_normal], Bc_fit[mask_normal],
         'o', color='steelblue', ms=8, zorder=5, label='Messwerte $B_c$')

if mask_anomal.any():
    ax2.plot(temps_fit[mask_anomal], Bc_fit[mask_anomal],
             's', color='steelblue', ms=8, markerfacecolor='none',
             markeredgewidth=1.5, zorder=5, label=r'Anomal: $T > T_c$ (Fit)')

# Untere Schranke
for T, B_mT, V_arr, Bc, status in all_data:
    if status == 'no_transition':
        ax2.plot(T, B_mT[-1], 'v', color='steelblue', ms=10, alpha=0.55, zorder=4)

# Fit-Kurve: nur im Bereich der Daten
T_data_min = min(T for T, *_ in all_data if _[-1] == 'ok')
T_fit_arr  = np.linspace(T_data_min, Tc_fit, 300)
Bc_fit_arr = bc_parabola(T_fit_arr, Bc0_fit, Tc_fit)
ax2.plot(T_fit_arr, Bc_fit_arr, '-', color='tomato', lw=2.2,
         label=f'Fit: $B_c(0)={Bc0_fit:.0f}$ mT, $T_c={Tc_fit:.3f}$ K')
ax2.axvline(Tc_fit, ls='--', color='tomato', alpha=0.4, lw=1.2)
ax2.text(Tc_fit + 0.02, max(Bc_fit) * 0.05,
         f'$T_c = {Tc_fit:.3f}$ K', color='tomato', fontsize=9, va='bottom')

# Pb-Bulk: mit Annotation sichtbar machen
T_bulk = np.linspace(4.5, 7.2, 200)
Bc_bulk = 80 * (1 - (T_bulk / 7.2)**2)
ax2.plot(T_bulk, Bc_bulk, ':', color='k', alpha=0.6, lw=2.0)
ax2.annotate('Pb Bulk\n(Lit.)', xy=(6.0, 80*(1-(6.0/7.2)**2)), xytext=(5.8, 180),
             fontsize=8, color='k', alpha=0.7,
             arrowprops=dict(arrowstyle='->', color='k', alpha=0.5, lw=0.8))

# Legende
from matplotlib.lines import Line2D
leg_els = [
    Line2D([0],[0], marker='o', color='steelblue', ms=8, lw=0, label='Messwerte $B_c$'),
    Line2D([0],[0], marker='s', color='steelblue', ms=8, lw=0, markerfacecolor='none',
           markeredgewidth=1.5, label=r'anomal ($T > T_c$ aus Fit)'),
    Line2D([0],[0], marker='v', color='steelblue', ms=10, lw=0, alpha=0.6,
           label=r'$B_c > B_\mathrm{max}$ (Schranke)'),
    Line2D([0],[0], color='tomato', lw=2.2, label=f'parabolischer Fit'),
]
ax2.legend(handles=leg_els, fontsize=9)
ax2.set_xlabel('$T$ (K)', fontsize=13)
ax2.set_ylabel(r'$\mu_0 H_c$ (mT)', fontsize=13)
ax2.set_title('Phasendiagramm $B_c(T)$ — Pb-Film (100 nm)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5.6, 7.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.pdf"), dpi=150)
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.png"), dpi=150)
print("Gespeichert: fig_bc_phase")
print("\nFertig.")
