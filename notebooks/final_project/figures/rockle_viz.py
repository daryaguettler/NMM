"""Generate a plot of the analytic Röckle-style temperature field with zone overlays.

This replaces the hand-drawn SVG schematic with an actual evaluation of the
closed-form model, which is both more informative and more honest. Uses
matplotlib for portable PNG output.
"""
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Problem setup matching the project spec
# ---------------------------------------------------------------------------
X_MIN, X_MAX = 0.0, 60.0
Z_MIN, Z_MAX = 0.0, 30.0
BLDG_X_MIN, BLDG_X_MAX = 25.0, 35.0
BLDG_Z_MIN, BLDG_Z_MAX = 0.0, 10.0

U_REF = 5.0          # m/s at z_ref
Z_REF = 10.0         # m
Z_0 = 0.5            # roughness length, m
T_REF = 25.0         # °C
T_FACADE_HOT = 50.0  # °C

NX, NZ = 200, 100
x = np.linspace(X_MIN, X_MAX, NX)
z = np.linspace(Z_MIN, Z_MAX, NZ)
X, Z = np.meshgrid(x, z, indexing="ij")

# ---------------------------------------------------------------------------
# Zone-by-zone temperature, following the spec §2.3
# ---------------------------------------------------------------------------
T = np.full_like(X, T_REF)

# Building mask
bldg_mask = (X >= BLDG_X_MIN) & (X <= BLDG_X_MAX) & (Z >= BLDG_Z_MIN) & (Z <= BLDG_Z_MAX)

# ----- Wall plume zone -----
# Schmidt-Linke scaling: T(x_wall + xi, z) - T_ref = (T_facade - T_ref) * exp(-xi / delta(z))
# delta(z) = c * z^(1/4) for laminar natural convection
# Plume exists immediately upwind of the heated facade (xi >= 0 from the wall)
xi = BLDG_X_MIN - X  # horizontal distance upwind of the heated wall (positive = upwind)
delta = 0.4 * np.maximum(Z, 0.5) ** 0.25  # boundary layer thickness, growing with height
plume_mask = (xi >= 0) & (xi < 5) & (Z >= BLDG_Z_MIN) & (Z <= BLDG_Z_MAX)
T_plume = T_REF + (T_FACADE_HOT - T_REF) * np.exp(-xi / delta)
T = np.where(plume_mask, T_plume, T)

# ----- Wake zone -----
# Gaussian deficit advecting warm wake from the building
# Width grows linearly with downwind distance; warming decays with distance
wake_mask = (X > BLDG_X_MAX) & (X < BLDG_X_MAX + 15)
x_downwind = X - BLDG_X_MAX
sigma_z = 2.0 + 0.3 * x_downwind          # wake grows downwind
z_c = BLDG_Z_MAX                           # center of wake at top of building
wake_decay = np.exp(-x_downwind / 8.0)     # warming decays with downwind distance
wake_lateral = np.exp(-((Z - z_c) ** 2) / (2 * sigma_z ** 2))
T_wake = T_REF + 6.0 * wake_decay * wake_lateral
T = np.where(wake_mask, np.maximum(T, T_wake), T)

# ----- Lifted plume above the heated wall -----
# Warm air rises above the heated facade as a thermal column
above_bldg = (X >= BLDG_X_MIN - 1) & (X <= BLDG_X_MIN + 4) & (Z >= BLDG_Z_MAX) & (Z < BLDG_Z_MAX + 8)
lift_decay = np.exp(-(Z - BLDG_Z_MAX) / 4.0) * np.exp(-((X - BLDG_X_MIN - 1.5) ** 2) / 8.0)
T = np.where(above_bldg, T + 8.0 * lift_decay, T)

# Building interior: mask
T = np.where(bldg_mask, np.nan, T)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))

# Temperature field
im = ax.pcolormesh(X, Z, T, shading='auto', cmap='RdBu_r',
                   vmin=T_REF, vmax=T_FACADE_HOT)
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
cbar.set_label('Temperature (°C)', fontsize=10)

# Zone overlays (semi-transparent rectangles with labels)
zones = [
    {'x': 0,  'z': 0,  'w': 20, 'h': 30, 'color': '#1f77b4', 'alpha': 0.08,
     'label': 'Upwind ABL zone', 'lx': 10, 'lz': 26},
    {'x': 20, 'z': 0,  'w': 5,  'h': 30, 'color': '#9b59b6', 'alpha': 0.10,
     'label': 'Displacement', 'lx': 22.5, 'lz': 26, 'rotate': 90, 'fontsize': 8},
    {'x': 35, 'z': 0,  'w': 15, 'h': 30, 'color': '#e67e22', 'alpha': 0.10,
     'label': 'Wake', 'lx': 42.5, 'lz': 26},
    {'x': 50, 'z': 0,  'w': 10, 'h': 30, 'color': '#1f77b4', 'alpha': 0.05,
     'label': 'Recovery → ABL', 'lx': 55, 'lz': 26},
]
for zone in zones:
    rect = Rectangle((zone['x'], zone['z']), zone['w'], zone['h'],
                     facecolor=zone['color'], alpha=zone['alpha'],
                     edgecolor=zone['color'], linewidth=1, linestyle='--')
    ax.add_patch(rect)
    rot = zone.get('rotate', 0)
    fontsize = zone.get('fontsize', 10)
    txt = ax.text(zone['lx'], zone['lz'], zone['label'],
                  ha='center', va='center', fontsize=fontsize,
                  fontweight='bold', color='#222', rotation=rot)
    txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='white')])

# Wall plume zone label (separate because it's small and special)
wall_plume_lbl = ax.text(27, 12, 'Wall plume', ha='center', va='center',
                          fontsize=9, fontweight='bold', color='#c4322f')
wall_plume_lbl.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='white')])
# Leader line to the plume zone
ax.annotate('', xy=(25.5, 8), xytext=(27, 11.2),
            arrowprops=dict(arrowstyle='-', color='#c4322f', lw=1))

# Building
bldg = Rectangle((BLDG_X_MIN, BLDG_Z_MIN), BLDG_X_MAX - BLDG_X_MIN, BLDG_Z_MAX - BLDG_Z_MIN,
                 facecolor='#555555', edgecolor='#222', linewidth=1.5)
ax.add_patch(bldg)
ax.text((BLDG_X_MIN + BLDG_X_MAX) / 2, (BLDG_Z_MIN + BLDG_Z_MAX) / 2, 'Building',
        ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Heated facade (red strip)
ax.plot([BLDG_X_MIN, BLDG_X_MIN], [BLDG_Z_MIN, BLDG_Z_MAX],
        color='#c4322f', linewidth=4, solid_capstyle='butt')
# Sun-heated annotation
ax.annotate('Sun-heated\nfacade', xy=(BLDG_X_MIN, 5), xytext=(15, 13),
            fontsize=9, color='#c4322f',
            arrowprops=dict(arrowstyle='->', color='#c4322f', lw=1.2),
            ha='center')

# Wind arrows at the inlet (showing ABL log profile direction)
for z_arrow in [4, 10, 18, 26]:
    # ABL log-law speed at this height for arrow length scaling
    u_star = U_REF * 0.4 / np.log((Z_REF + Z_0) / Z_0)
    U_at_z = u_star / 0.4 * np.log((z_arrow + Z_0) / Z_0)
    arrow_len = 1.5 + 0.4 * U_at_z
    ax.annotate('', xy=(arrow_len, z_arrow), xytext=(0.5, z_arrow),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))
ax.text(7, 28, 'Wind →   U(z) ~ log-law', fontsize=10, color='#1f77b4', fontweight='bold')

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Z_MIN, Z_MAX)
ax.set_aspect('equal')
ax.set_xlabel('Along-wind distance x (m)', fontsize=11)
ax.set_ylabel('Height z (m)', fontsize=11)
ax.set_title('Method 1: Analytic Röckle-style zonification', fontsize=13, fontweight='bold')
ax.tick_params(labelsize=9)
ax.set_facecolor('#fafafa')

plt.tight_layout()
output_dir = Path(__file__).resolve().parent
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "method1_rockle_zones.png"
plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor='white')
plt.close()
print(f"wrote {output_path}")
