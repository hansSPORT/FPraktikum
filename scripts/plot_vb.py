"""
V(B) Messungen — Bleifilm Bc(T) Phasendiagramm
Pb-Film (100 nm), He-Kryostat, 4-Punkt-Messung

B-Feld Rekonstruktion: B(t) = 5.262 mT/s * t
(B-Kanal im Mess-Skript auskommentiert, Rate aus Magnet-Parametern berechnet:
 rate=0.05 A/s, conversion=0.10524 T/A → B-Rate = 5.262 mT/s)

Besonderheiten:
- T=5.821K: kein Übergang sichtbar (Bc > B_max ≈ 505 mT) → untere Schranke
- T=5.916/6.012/6.110K: Sweep bis 1000 mT (191 Datenpunkte, B_stop=10 kG)
- T>=6.796K: Film schon normal bei B=0 (über Tc des Films)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os
import re

DATA_DIR = "/home/user/Uni-Projekte/privat/Fpraktikum/Praktikumsdaten Sortiert/V-B"
FIG_DIR = "/home/user/FPraktikum/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# B-Feld-Rate: rate=0.05 A/s, conversion=0.10524 T/A
B_RATE_mT_per_s = 0.05 * 0.10524 * 1000  # = 5.262 mT/s
print(f"B-Sweep-Rate: {B_RATE_mT_per_s:.4f} mT/s")

# Spannungsschwellen für Transition-Erkennung
V_SC_MAX = 2e-3    # V: Film gilt als SC wenn V < 2 mV
V_N_MIN  = 3e-3    # V: Film gilt als normal wenn V > 3 mV
V_RANGE_MIN = 1e-3 # V: Mindest-V-Bereich für echte Transition

def load_vb_file(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    t = float(parts[0])
                    v = float(parts[2])
                    i = float(parts[4])
                    data.append((t, v, i))
                except ValueError:
                    continue
    return np.array(data)

def extract_temperature(filename):
    m = re.search(r'data_[+]?([0-9]+\.[0-9]+)', os.path.basename(filename))
    return float(m.group(1)) if m else None

def find_bc_50percent(B_mT, V):
    """Bc via 50%-Schwellwert; gibt (Bc, status) zurück.
    Status: 'ok', 'no_transition' (immer SC), 'already_normal' (immer N)
    """
    V_start = np.median(V[:8])
    V_end   = np.median(V[-8:])
    # Erkenne Zustand
    if V_start > V_N_MIN and V_end > V_N_MIN:
        return None, 'already_normal'
    if V_start < V_SC_MAX and V_end < V_SC_MAX:
        return None, 'no_transition'
    if (V_end - V_start) < V_RANGE_MIN:
        return None, 'no_transition'
    # Echte Transition
    V_thresh = 0.5 * (V_start + V_end)
    idx = np.where(V > V_thresh)[0]
    if len(idx) == 0:
        return None, 'no_transition'
    i = idx[0]
    if i == 0:
        return B_mT[0], 'ok'
    frac = (V_thresh - V[i-1]) / (V[i] - V[i-1])
    Bc = B_mT[i-1] + frac * (B_mT[i] - B_mT[i-1])
    return Bc, 'ok'

# --- Lade alle V-B Dateien ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.dat")))
print(f"Gefundene Dateien: {len(files)}")

all_data = []       # (T, B_mT, V_arr, Bc, status)
temps_fit = []
Bc_fit = []

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
    status_str = {'ok': 'Übergang', 'no_transition': 'kein Übergang (Bc > B_max)',
                  'already_normal': 'schon normal bei B=0'}[status]
    if status == 'ok':
        temps_fit.append(T)
        Bc_fit.append(Bc)
        print(f"  T={T:.3f} K  Bc={Bc:.1f} mT  [{status_str}]")
    else:
        Bmax = B_mT[-1]
        print(f"  T={T:.3f} K  [{status_str}]  B_max={Bmax:.0f} mT")

temps_fit = np.array(temps_fit)
Bc_fit = np.array(Bc_fit)

print(f"\nPunkte für Fit: {len(temps_fit)}")

# --- Bc(T) Fit ---
def bc_parabola(T, Bc0, Tc):
    return Bc0 * (1 - (T / Tc)**2)

try:
    p0 = [1200.0, 6.75]
    popt, pcov = curve_fit(bc_parabola, temps_fit, Bc_fit, p0=p0,
                           bounds=([100, 5.0], [5000, 9.0]))
    perr = np.sqrt(np.diag(pcov))
    Bc0_fit, Tc_fit = popt
    Bc0_err, Tc_err = perr
    print(f"\nFit-Ergebnis:")
    print(f"  Bc(0) = {Bc0_fit:.1f} ± {Bc0_err:.1f} mT")
    print(f"  Tc    = {Tc_fit:.4f} ± {Tc_err:.4f} K")
    fit_ok = True
except Exception as e:
    print(f"Fit fehlgeschlagen: {e}")
    fit_ok = False
    Tc_fit, Bc0_fit = 7.0, 1000

# --- Figure 1: V(B) Kurven ---
fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
cmap = matplotlib.colormaps['coolwarm']
T_all = [d[0] for d in all_data]
T_min, T_max_val = min(T_all), max(T_all)

for T, B_mT, V_arr, Bc, status in all_data:
    color = cmap((T - T_min) / (T_max_val - T_min))
    ls = '-' if status == 'ok' else ':'
    alpha = 0.9 if status == 'ok' else 0.5
    ax1.plot(B_mT, V_arr * 1e3, color=color, lw=1.3, ls=ls, alpha=alpha)
    if status == 'ok' and Bc is not None:
        Bc_V = np.interp(Bc, B_mT, V_arr)
        ax1.plot(Bc, Bc_V * 1e3, 'o', color=color, ms=6,
                 markeredgecolor='k', markeredgewidth=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_min, T_max_val))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1)
cbar.set_label('$T$ (K)', fontsize=12)
ax1.set_xlabel('$\\mu_0 H$ (mT)', fontsize=13)
ax1.set_ylabel('$U$ (mV)', fontsize=13)
ax1.set_title('Spannungstransition Pb-Film: $U(B)$ bei verschiedenen Temperaturen', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
# Legende-Einträge für Kurven-Typen
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color='gray', lw=1.3, ls='-', label='SC→N Übergang sichtbar'),
    Line2D([0], [0], color='gray', lw=1.3, ls=':', alpha=0.5, label='kein Übergang / schon normal'),
    Line2D([0], [0], marker='o', color='gray', ms=6, lw=0, markeredgecolor='k',
           label='$B_c$ (50%-Schwelle)'),
]
ax1.legend(handles=legend_els, fontsize=9, loc='upper left')
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "fig_vb_curves.pdf"), dpi=150)
fig1.savefig(os.path.join(FIG_DIR, "fig_vb_curves.png"), dpi=150)
print("Gespeichert: fig_vb_curves.pdf/.png")

# --- Figure 2: Bc(T) Phasendiagramm ---
fig2, ax2 = plt.subplots(figsize=(7, 5))

# Punkte ohne Übergang (kein Übergang → Bc > B_max, untere Schranke)
for T, B_mT, V_arr, Bc, status in all_data:
    if status == 'no_transition':
        ax2.plot(T, B_mT[-1], 'v', color='steelblue', ms=10, alpha=0.6, zorder=4)

# Messwerte mit Übergang
ax2.plot(temps_fit, Bc_fit, 'o', color='steelblue', ms=8, zorder=5, label='Messwerte $B_c$')

# Fit-Kurve
if fit_ok:
    T_plot = np.linspace(4.5, Tc_fit, 300)
    Bc_plot = bc_parabola(T_plot, Bc0_fit, Tc_fit)
    ax2.plot(T_plot, Bc_plot, '-', color='tomato', lw=2.2,
             label=f'Parabolischer Fit\n$B_c(0)={Bc0_fit:.0f}\\pm{Bc0_err:.0f}$ mT,  $T_c={Tc_fit:.3f}\\pm{Tc_err:.4f}$ K')
    ax2.axvline(Tc_fit, ls='--', color='tomato', alpha=0.4, lw=1)

# Pb-Bulk-Referenz
T_bulk = np.linspace(0, 7.2, 200)
Bc_bulk = 80 * (1 - (T_bulk / 7.2)**2)
ax2.plot(T_bulk, Bc_bulk, ':', color='k', alpha=0.5, lw=1.8, label='Pb Bulk (Lit.)')

# Legende für Symbole
from matplotlib.lines import Line2D
leg_els = [
    Line2D([0],[0], marker='o', color='steelblue', ms=8, lw=0, label='Messwerte $B_c$'),
    Line2D([0],[0], marker='v', color='steelblue', ms=10, lw=0, alpha=0.6, label='$B_c > B_\\mathrm{max}$ (untere Schranke)'),
    Line2D([0],[0], color='tomato', lw=2.2, label=f'Fit: $B_c(0)={Bc0_fit:.0f}$ mT, $T_c={Tc_fit:.3f}$ K'),
    Line2D([0],[0], color='k', lw=1.8, ls=':', alpha=0.5, label='Pb Bulk (Lit.)'),
]
ax2.legend(handles=leg_els, fontsize=9)
ax2.set_xlabel('$T$ (K)', fontsize=13)
ax2.set_ylabel('$\\mu_0 H_c$ (mT)', fontsize=13)
ax2.set_title('Phasendiagramm $B_c(T)$ — Pb-Film (100 nm)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5.2, 7.6)
ax2.set_ylim(bottom=0)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.pdf"), dpi=150)
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.png"), dpi=150)
print("Gespeichert: fig_bc_phase.pdf/.png")
print("\nFertig.")
