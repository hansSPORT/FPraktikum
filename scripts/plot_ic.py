"""
I-V Messungen — Kritischer Strom Ic(T)
Pb-Film (100 nm), He-Kryostat, 4-Punkt-Messung
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import glob, os, re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

DATA_DIR = "/home/user/Uni-Projekte/privat/Fpraktikum/Praktikumsdaten Sortiert/I_c(T)"
FIG_DIR  = "/home/user/FPraktikum/figures"
os.makedirs(FIG_DIR, exist_ok=True)

V_OFFSET_THRESHOLD = 500e-6  # 500 µV über Basislinie

def load_iv_file(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 2:
                try:
                    data.append((float(p[0]), float(p[1])))
                except ValueError:
                    continue
    return np.array(data)

def extract_temperature(filename):
    m = re.search(r'data_[+]?([0-9]+\.[0-9]+)', os.path.basename(filename))
    return float(m.group(1)) if m else None

def find_ic(I, V):
    V_base = np.median(V[:5])
    thresh = V_base + V_OFFSET_THRESHOLD
    idx = np.where(V > thresh)[0]
    if len(idx) == 0:
        return None, V_base, 'no_transition'
    i = idx[0]
    if i == 0:
        return I[0], V_base, 'ok'
    frac = (thresh - V[i-1]) / (V[i] - V[i-1])
    return I[i-1] + frac * (I[i] - I[i-1]), V_base, 'ok'

# --- Laden ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.dat")))
all_data = []
temps_fit, Ic_fit_uA = [], []

for f in files:
    T = extract_temperature(f)
    if T is None:
        continue
    data = load_iv_file(f)
    if len(data) < 3:
        continue
    V_arr = data[:, 0]
    I_arr = data[:, 1]
    mask = I_arr >= 0
    I_pos = I_arr[mask]
    V_pos = V_arr[mask]
    idx_sort = np.argsort(I_pos)
    I_pos = I_pos[idx_sort]
    V_pos = V_pos[idx_sort]
    Ic, V_base, status = find_ic(I_pos, V_pos)
    all_data.append((T, I_pos, V_pos, Ic, status, V_base))
    if status == 'ok':
        temps_fit.append(T)
        Ic_fit_uA.append(Ic * 1e6)

temps_fit  = np.array(temps_fit)
Ic_fit_uA  = np.array(Ic_fit_uA)

# ======================================================
# FIGURE 1: I-V Kennlinien mit Inset
# Fix: Hauptplot zeigt gesamten Bereich, Inset zeigt
#      den Übergangsbereich (0 bis ~1.5x max(Ic)) gezoomt
# ======================================================
fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
cmap = matplotlib.colormaps['plasma']
T_all     = [d[0] for d in all_data]
T_min_c, T_max_c = min(T_all), max(T_all)

# --- Inset vorbereiten ---
ax_in = inset_axes(ax1, width='42%', height='45%', loc='upper left',
                   bbox_to_anchor=(0.08, 0.02, 1, 1),
                   bbox_transform=ax1.transAxes)

Ic_max_mA = max(Ic_fit_uA) / 1000 * 1.35  # etwas über dem größten Ic

for T, I_pos, V_pos, Ic, status, V_base in all_data:
    color = cmap((T - T_min_c) / (T_max_c - T_min_c))
    DV = (V_pos - V_base) * 1e3  # mV
    lw = 1.5 if status == 'ok' else 1.0
    ax1.plot(I_pos * 1e3, DV, color=color, lw=lw, alpha=0.9)

    # Inset: nur bis Ic_max_mA
    mask_in = I_pos * 1e3 <= Ic_max_mA
    if mask_in.any():
        ax_in.plot(I_pos[mask_in] * 1e3, DV[mask_in], color=color, lw=1.5, alpha=0.9)

    if status == 'ok' and Ic is not None:
        Ic_mA = Ic * 1e3
        DV_ic = (np.interp(Ic, I_pos, V_pos) - V_base) * 1e3
        ax1.plot(Ic_mA, DV_ic, 'o', color=color, ms=6,
                 markeredgecolor='k', markeredgewidth=0.7, zorder=5)
        if Ic_mA <= Ic_max_mA:
            ax_in.plot(Ic_mA, DV_ic, 'o', color=color, ms=5,
                       markeredgecolor='k', markeredgewidth=0.7, zorder=5)

# Schwelllinie im Inset
ax_in.axhline(V_OFFSET_THRESHOLD * 1e3, ls='--', color='gray', lw=1.2,
              label='+500 µV Schwelle')
ax_in.set_xlabel('$I$ (mA)', fontsize=9)
ax_in.set_ylabel(r'$\Delta U$ (mV)', fontsize=9)
ax_in.set_xlim(0, Ic_max_mA)
ax_in.set_ylim(-0.05, 2.5)
ax_in.tick_params(labelsize=8)
ax_in.grid(True, alpha=0.3)
ax_in.set_title('Übergangsbereich', fontsize=8, pad=3)
ax_in.legend(fontsize=7, loc='upper left')

# Rechteck im Hauptplot markieren
rect = matplotlib.patches.Rectangle((0, -0.5), Ic_max_mA, 3.2,
                                     linewidth=1, edgecolor='gray',
                                     facecolor='lightyellow', alpha=0.35, zorder=0)
ax1.add_patch(rect)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_min_c, T_max_c))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1)
cbar.set_label('$T$ (K)', fontsize=12)

ax1.set_xlabel('$I$ (mA)', fontsize=13)
ax1.set_ylabel(r'$\Delta U$ (mV)  [über SC-Basislinie]', fontsize=12)
ax1.set_title('I-V-Kennlinien Pb-Film — Übergang bei $I_c(T)$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=-0.5)
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "fig_iv_curves.pdf"), dpi=150)
fig1.savefig(os.path.join(FIG_DIR, "fig_iv_curves.png"), dpi=150)
print("Gespeichert: fig_iv_curves")

# ======================================================
# FIGURE 2: Ic(T)
# Fix: Fit nur in Datenbereich, kein wilder Extrapolations-
#      schwanz nach links; x-Achse eng ans Datenfenster
# ======================================================
Tc_fixed = 6.8606  # aus Bc(T)-Fit

def ic_gl(T, Ic0, n):
    return Ic0 * np.clip(1 - T / Tc_fixed, 0, None)**n

popt, pcov = curve_fit(ic_gl, temps_fit, Ic_fit_uA, p0=[5e5, 1.5],
                       bounds=([0, 0.1], [1e9, 10]))
perr = np.sqrt(np.diag(pcov))
print(f"Ic(T)-Fit: n = {popt[1]:.3f} ± {perr[1]:.3f}")

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

# Untere Schranke
for T, I_pos, V_pos, Ic, status, V_base in all_data:
    if status == 'no_transition':
        ax2.plot(T, I_pos[-1]*1e6, 'v', color='darkorchid', ms=9,
                 markerfacecolor='none', markeredgewidth=1.8,
                 alpha=0.8, zorder=4)

# Messwerte
ax2.plot(temps_fit, Ic_fit_uA, 'o', color='darkorchid', ms=8, zorder=5)

# Fit: NUR im Datenbereich (min(T_data) bis Tc)
T_fit_range = np.linspace(min(temps_fit), Tc_fixed, 200)
Ic_fit_range = ic_gl(T_fit_range, *popt)
ax2.plot(T_fit_range, Ic_fit_range, '--', color='tomato', lw=2,
         label=f'Fit ($T_c$ fixiert $={Tc_fixed:.3f}$ K, $n={popt[1]:.2f}$)')
ax2.axvline(Tc_fixed, ls='--', color='tomato', alpha=0.4, lw=1.2)
ax2.text(Tc_fixed + 0.005, 100, f'$T_c={Tc_fixed:.3f}$ K',
         color='tomato', fontsize=9)

leg_els = [
    Line2D([0],[0], marker='o', color='darkorchid', ms=8, lw=0,
           label='Messwerte $I_c$'),
    Line2D([0],[0], marker='v', color='darkorchid', ms=9, lw=0,
           markerfacecolor='none', markeredgewidth=1.8, alpha=0.8,
           label=r'$I_c > I_\mathrm{max}$ (Schranke)'),
    Line2D([0],[0], color='tomato', lw=2, ls='--',
           label=f'Fit: $n={popt[1]:.2f}\\pm{perr[1]:.2f}$'),
]
ax2.legend(handles=leg_els, fontsize=10)
ax2.set_xlabel('$T$ (K)', fontsize=13)
ax2.set_ylabel('$I_c$ (µA)', fontsize=13)
ax2.set_title('Kritischer Strom $I_c(T)$ — Pb-Film (100 nm)', fontsize=12)
ax2.grid(True, alpha=0.3)
# x-Achse eng: nur Datenbereich + kleiner Rand
ax2.set_xlim(min(temps_fit) - 0.05, Tc_fixed + 0.05)
ax2.set_ylim(bottom=0)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_ic_temp.pdf"), dpi=150)
fig2.savefig(os.path.join(FIG_DIR, "fig_ic_temp.png"), dpi=150)
print("Gespeichert: fig_ic_temp")
print("\nFertig.")
