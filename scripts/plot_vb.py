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
V_SC_MAX      = 2e-3   # V — Film gilt als SC wenn V_start < 2 mV
V_ALREADY_N   = 5.5e-3 # V — Film gilt bei B=0 schon als normal wenn V_start > 5.5 mV
V_N_COMPLETE  = 5.5e-3 # V — Sweep gilt als "vollständig bis Normal" wenn V_end > 5.5 mV

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

# --- Laden (Schritt 1): V_N_global aus vollständigen Messungen bestimmen ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.dat")))
raw = []
for f in files:
    T = extract_temperature(f)
    if T is None:
        continue
    data = load_vb_file(f)
    if len(data) < 5:
        continue
    raw.append((T, data))

V_N_values = []
for T, data in raw:
    V_start = np.median(data[:8, 1])
    V_end   = np.median(data[-8:, 1])
    # Vollständige SC→N Transition: SC am Anfang, vollständig normal am Ende
    if V_start < V_SC_MAX and V_end > V_N_COMPLETE:
        V_N_values.append(np.median(data[-4:, 1]))  # letzten 4 Punkte stabiler

V_N_global = np.median(V_N_values)
print(f"V_N_global = {V_N_global*1e6:.0f} µV  (aus {len(V_N_values)} vollständigen Messungen)")

def find_bc_50percent(B_mT, V, V_N_ref):
    """50%-Schwelle mit festem V_N_ref (globalem Normalzustandswert)."""
    V_start = np.median(V[:5])

    # Schon normal bei B=0
    if V_start > V_ALREADY_N:
        return None, 'already_normal'

    # 50%-Schwelle mit globalem V_N
    V_thresh = 0.5 * (V_start + V_N_ref)

    idx = np.where(V > V_thresh)[0]
    if len(idx) == 0:
        # Threshold nicht erreicht — untere Schranke
        return None, 'no_transition'
    i = idx[0]
    if i == 0:
        return B_mT[0], 'ok'
    frac = (V_thresh - V[i-1]) / (V[i] - V[i-1])
    return B_mT[i-1] + frac * (B_mT[i] - B_mT[i-1]), 'ok'

# --- Laden (Schritt 2): Bc bestimmen ---
all_data = []
temps_fit, Bc_fit = [], []

for T, data in raw:
    t_arr = data[:, 0]
    V_arr = data[:, 1]
    B_mT  = B_RATE_mT_per_s * t_arr
    Bc, status = find_bc_50percent(B_mT, V_arr, V_N_global)
    V_end_meas = np.median(V_arr[-4:])
    # Kurve: vollständig transitiert wenn V_end_meas > 90% V_N_global
    complete = V_end_meas > 0.90 * V_N_global
    all_data.append((T, B_mT, V_arr, Bc, status, complete))
    marker = f"Bc={Bc:.1f} mT" if Bc else "—"
    print(f"  T={T:.3f} K  V_end={V_end_meas*1e6:.0f}µV  complete={complete}  {status}  {marker}")
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

for T, B_mT, V_arr, Bc, status, complete in all_data:
    color = cmap((T - T_min_c) / (T_max_c - T_min_c))
    if status == 'ok':
        # Plateau nur 150 mT hinter Bc zeigen — verhindert Überlagerungsartefakt
        B_end = (Bc + 150) if Bc is not None else B_mT[-1]
        mask = B_mT <= B_end
        ax1.plot(B_mT[mask], V_arr[mask] * 1e3, color=color, lw=1.5, alpha=0.9, zorder=3)
        if Bc is not None:
            Bc_V = np.interp(Bc, B_mT, V_arr)
            ax1.plot(Bc, Bc_V * 1e3, 'o', color=color, ms=6,
                     markeredgecolor='k', markeredgewidth=0.7, zorder=5)
    elif status == 'no_transition':
        ax1.plot(B_mT, V_arr * 1e3, color=color, lw=1.2, ls='--', alpha=0.75, zorder=2)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_min_c, T_max_c))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1)
cbar.set_label('$T$ (K)', fontsize=12)

legend_els = [
    Line2D([0],[0], color='dimgray', lw=1.5, label='SC→N Übergang'),
    Line2D([0],[0], color='dimgray', lw=1.2, ls='--', alpha=0.75,
           label=r'kein Übergang ($B_c > B_\mathrm{max}$)'),
    Line2D([0],[0], marker='o', color='dimgray', ms=6, lw=0,
           markeredgecolor='k', label='$B_c$ (50%-Schwelle)'),
]
ax1.legend(handles=legend_els, fontsize=9, loc='upper left')
ax1.set_xlabel(r'$\mu_0 H$ (mT)', fontsize=13)
ax1.set_ylabel('$U$ (mV)', fontsize=13)
ax1.set_title('Spannungstransition Pb-Film: $U(B)$ bei verschiedenen Temperaturen', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1060)
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

# Alle Messpunkte als einheitliche Kreise (Anomalie T>Tc wird im Text diskutiert)
ax2.plot(temps_fit, Bc_fit,
         'o', color='steelblue', ms=8, zorder=5)

# Untere Schranke (aufwärts-Dreieck: wahrer Bc liegt ÜBER dem Marker)
for T, B_mT, V_arr, Bc, status, complete in all_data:
    if status == 'no_transition':
        ax2.plot(T, B_mT[-1], '^', color='steelblue', ms=10, alpha=0.75, zorder=4)

# Fit-Kurve: nur im Bereich der Daten
T_data_min = min(T for T, B_mT, V_arr, Bc, status, complete in all_data if status == 'ok')
T_fit_arr  = np.linspace(T_data_min, Tc_fit, 300)
Bc_fit_arr = bc_parabola(T_fit_arr, Bc0_fit, Tc_fit)
ax2.plot(T_fit_arr, Bc_fit_arr, '-', color='tomato', lw=2.2,
         label=f'Fit: $B_c(0)={Bc0_fit:.0f}$ mT, $T_c={Tc_fit:.3f}$ K')
ax2.axvline(Tc_fit, ls='--', color='tomato', alpha=0.4, lw=1.2)
ax2.text(Tc_fit + 0.02, max(Bc_fit) * 0.05,
         f'$T_c = {Tc_fit:.3f}$ K', color='tomato', fontsize=9, va='bottom')

# Pb-Bulk: gepunktet, mit Label direkt an der Linie
T_bulk = np.linspace(4.5, 7.0, 200)
Bc_bulk = 80 * (1 - (T_bulk / 7.2)**2)
ax2.plot(T_bulk, Bc_bulk, '--', color='dimgray', alpha=0.75, lw=1.8)
T_lbl = 5.75
ax2.text(T_lbl, 80*(1-(T_lbl/7.2)**2) + 22, 'Pb Bulk (Lit.)',
         fontsize=8, color='dimgray', alpha=0.9, va='bottom')

# Legende
from matplotlib.lines import Line2D
leg_els = [
    Line2D([0],[0], marker='o', color='steelblue', ms=8, lw=0, label='Messwerte $B_c$'),
    Line2D([0],[0], marker='^', color='steelblue', ms=10, lw=0, alpha=0.75,
           label=r'$B_c > B_\mathrm{max}$ (untere Schranke)'),
    Line2D([0],[0], color='tomato', lw=2.2, label=f'parabolischer Fit'),
]
ax2.legend(handles=leg_els, fontsize=9)
ax2.set_xlabel('$T$ (K)', fontsize=13)
ax2.set_ylabel(r'$\mu_0 H_c$ (mT)', fontsize=13)
ax2.set_title('Phasendiagramm $B_c(T)$ — Pb-Film (100 nm)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(5.6, 7.0)
ax2.set_ylim(bottom=0)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.pdf"), dpi=150)
fig2.savefig(os.path.join(FIG_DIR, "fig_bc_phase.png"), dpi=150)
print("Gespeichert: fig_bc_phase")
print("\nFertig.")
