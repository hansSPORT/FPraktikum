"""
I-V Messungen — Kritischer Strom Ic(T)
Pb-Film (100 nm), He-Kryostat, 4-Punkt-Messung

Yokogawa GS200 als Spannungsquelle, R_pre=1kOhm → I = V_yoko/R_pre
Spannung V gemessen mit HP3458A, durch voltage_amp=10 geteilt
SC-Basislinie ~290 µV (Offset aus Kontaktwiderstand + Thermospannung)
Ic-Bestimmung: V > V_base + 500 µV (deutlich über Rauschen der Basislinie)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os
import re

DATA_DIR = "/home/user/Uni-Projekte/privat/Fpraktikum/Praktikumsdaten Sortiert/I_c(T)"
FIG_DIR = "/home/user/FPraktikum/figures"
os.makedirs(FIG_DIR, exist_ok=True)

V_OFFSET_THRESHOLD = 500e-6  # 500 µV über Basislinie = SC→N Übergang

def load_iv_file(path):
    """Lade I-V Datei; gibt (V, I) zurück (V in V, I in A)."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    V, I = float(parts[0]), float(parts[1])
                    data.append((V, I))
                except ValueError:
                    continue
    return np.array(data)

def extract_temperature(filename):
    m = re.search(r'data_[+]?([0-9]+\.[0-9]+)', os.path.basename(filename))
    return float(m.group(1)) if m else None

def find_ic(I, V):
    """Bestimme Ic: erster Strom, wo V mehr als 500 µV über SC-Basislinie liegt.
    Gibt (Ic, V_base, status) zurück; status: 'ok' oder 'no_transition'.
    """
    # SC-Basislinie = Median der ersten 5 Punkte (bei I≈0)
    V_base = np.median(V[:5])
    thresh = V_base + V_OFFSET_THRESHOLD
    # Suche ersten Punkt über Schwelle
    idx = np.where(V > thresh)[0]
    if len(idx) == 0:
        return None, V_base, 'no_transition'
    i = idx[0]
    if i == 0:
        return I[0], V_base, 'ok'
    # Lineare Interpolation
    frac = (thresh - V[i-1]) / (V[i] - V[i-1])
    Ic = I[i-1] + frac * (I[i] - I[i-1])
    return Ic, V_base, 'ok'

# --- Lade alle I-V Dateien ---
files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.dat")))
print(f"Gefundene Dateien: {len(files)}")

all_data = []    # (T, I_arr, V_arr, Ic, status, V_base)
temps_fit = []
Ic_fit_uA = []

for f in files:
    T = extract_temperature(f)
    if T is None:
        continue
    data = load_iv_file(f)
    if len(data) < 3:
        continue
    V_arr = data[:, 0]
    I_arr = data[:, 1]
    # Positiver Ast, nach Strom sortiert
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
        print(f"  T={T:.3f} K  →  Ic={Ic*1e6:.0f} µA   (V_base={V_base*1e6:.0f} µV)")
    else:
        Imax = I_pos[-1] * 1e6
        print(f"  T={T:.3f} K  →  kein Übergang (Ic > {Imax:.0f} µA)")

temps_fit = np.array(temps_fit)
Ic_fit_uA = np.array(Ic_fit_uA)
print(f"\nPunkte für Fit: {len(temps_fit)}")

# --- Figure 1: I-V Kennlinien ---
fig1, ax1 = plt.subplots(figsize=(8.5, 5.5))
cmap = matplotlib.colormaps['plasma']
T_all = [d[0] for d in all_data]
T_min, T_max_val = min(T_all), max(T_all)

for T, I_pos, V_pos, Ic, status, V_base in all_data:
    color = cmap((T - T_min) / (T_max_val - T_min))
    # Voltage abzüglich Basislinie für Übersichtlichkeit
    V_plot = (V_pos - V_base) * 1e3  # mV über Basislinie
    ax1.plot(I_pos * 1e3, V_plot, color=color, lw=1.4, alpha=0.85)
    if status == 'ok' and Ic is not None:
        V_ic = (np.interp(Ic, I_pos, V_pos) - V_base) * 1e3
        ax1.plot(Ic * 1e3, V_ic, 'o', color=color, ms=6,
                 markeredgecolor='k', markeredgewidth=0.5)

# Schwellwert-Linie
ax1.axhline(V_OFFSET_THRESHOLD * 1e3, ls='--', color='gray', lw=1.2,
            label=f'Schwelle +{V_OFFSET_THRESHOLD*1e6:.0f} µV')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(T_min, T_max_val))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1)
cbar.set_label('$T$ (K)', fontsize=12)
ax1.set_xlabel('$I$ (mA)', fontsize=13)
ax1.set_ylabel('$\\Delta U$ (mV)  [Über SC-Basislinie]', fontsize=12)
ax1.set_title('I-V Kennlinien Pb-Film — Übergang bei $I_c(T)$', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=-0.02)
plt.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "fig_iv_curves.pdf"), dpi=150)
fig1.savefig(os.path.join(FIG_DIR, "fig_iv_curves.png"), dpi=150)
print("Gespeichert: fig_iv_curves.pdf/.png")

# --- Figure 2: Ic(T) ---
fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

# Untere Schranken
for T, I_pos, V_pos, Ic, status, V_base in all_data:
    if status == 'no_transition':
        ax2.plot(T, I_pos[-1]*1e6, 'v', color='darkorchid', ms=9, alpha=0.5, zorder=4)

ax2.plot(temps_fit, Ic_fit_uA, 'o', color='darkorchid', ms=8, zorder=5)

# Fit: Ic ~ Ic0 * (1 - (T/Tc)^2)^n
if len(temps_fit) >= 3:
    def ic_power(T, Ic0, Tc, n):
        val = np.clip(1 - (T / Tc)**2, 0, None)
        return Ic0 * val**n

    # Tc aus Bc(T)-Fit fixiert: Tc=6.861K. Nur Ic0 und n anpassen.
    Tc_fixed = 6.8606  # aus Bc(T)-parabolischem Fit
    def ic_gl(T, Ic0, n):
        return Ic0 * np.clip(1 - T / Tc_fixed, 0, None)**n
    try:
        p0 = [5e5, 1.5]
        popt, pcov = curve_fit(ic_gl, temps_fit, Ic_fit_uA, p0=p0,
                               bounds=([0, 0.1], [1e9, 10]))
        perr = np.sqrt(np.diag(pcov))
        T_plot = np.linspace(min(temps_fit) - 0.2, Tc_fixed, 200)
        Ic_plot = ic_gl(T_plot, *popt)
        ax2.plot(T_plot, Ic_plot, '--', color='tomato', lw=2,
                 label=f'Fit ($T_c$ fixiert $={Tc_fixed:.3f}$ K):\n$n={popt[1]:.2f}\\pm{perr[1]:.2f}$')
        ax2.axvline(Tc_fixed, ls='--', color='tomato', alpha=0.4, lw=1)
        print(f"\nIc(T)-Fit (Tc fixiert = {Tc_fixed} K):")
        print(f"  Ic(0) = {popt[0]:.0f} ± {perr[0]:.0f} µA")
        print(f"  n     = {popt[1]:.3f} ± {perr[1]:.3f}")
    except Exception as e:
        print(f"Ic-Fit fehlgeschlagen: {e}")

from matplotlib.lines import Line2D
leg_els = [
    Line2D([0],[0], marker='o', color='darkorchid', ms=8, lw=0, label='Messwerte $I_c$'),
    Line2D([0],[0], marker='v', color='darkorchid', ms=9, lw=0, alpha=0.5, label='$I_c > I_\\mathrm{max}$ (untere Schranke)'),
    Line2D([0],[0], color='tomato', lw=2, ls='--', label='Fit'),
]
ax2.legend(handles=leg_els, fontsize=10)
ax2.set_xlabel('$T$ (K)', fontsize=13)
ax2.set_ylabel('$I_c$ (µA)', fontsize=13)
ax2.set_title('Kritischer Strom $I_c(T)$ — Pb-Film (100 nm)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_ic_temp.pdf"), dpi=150)
fig2.savefig(os.path.join(FIG_DIR, "fig_ic_temp.png"), dpi=150)
print("Gespeichert: fig_ic_temp.pdf/.png")
print("\nFertig.")
