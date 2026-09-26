"""Пересчитать коэффициенты и графики из сохранённых GPU-замеров."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, nnls
from equations import flops, memory, latency, energy

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'results'
(out / 'figures').mkdir(parents=True, exist_ok=True)
df = pd.read_csv(out / 'measurements.csv')
ok = df[df.status == 'OK'].copy()
s, b = ok.S.to_numpy(), ok.B.to_numpy()
fit_rows = ok.is_validation.to_numpy() == 0
f = flops(s, b) / 1e9
q = 4 * (1040324 + b * (91.0 * s**2 + 2148)) / 1e9
t = ok.latency_s.to_numpy()
f0, q0, t0 = f[fit_rows], q[fit_rows], t[fit_rows]
fits = [least_squares(lambda p: (p[0] + np.maximum(p[1]*f0, p[2]*q0) - t0) / t0,
                     [t0.min()/10, np.median(t0/f0)*r, np.median(t0/q0)/r],
                     bounds=(0, np.inf), x_scale='jac', max_nfev=5000) for r in [0.1, 1, 10]]
p = min((r for r in fits if r.success), key=lambda r: np.sum(r.fun**2)).x
theta = dict(launch_s=float(p[0]), seconds_per_flop=float(p[1]/1e9), seconds_per_byte=float(p[2]/1e9))
e = ok.energy_j.to_numpy()
A = np.column_stack([latency(s,b,theta), f, q])[fit_rows] / e[fit_rows,None]
scale = np.linalg.norm(A, axis=0)
c = nnls(A/scale, np.ones(fit_rows.sum()))[0] / scale
theta_energy = dict(base_power_w=float(c[0]), joules_per_flop=float(c[1]/1e9),
                    joules_per_byte=float(c[2]/1e9), latency=theta)
(out / 'theta.json').write_text(json.dumps(dict(latency=theta, energy=theta_energy), indent=2))
print(theta, theta_energy)

metrics = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name, unit, calc in zip(axes, ['latency_s', 'memory_bytes', 'energy_j'],
                              ['s', 'bytes', 'J'],
                              [latency(s,b,theta), memory(s,b), energy(s,b,theta_energy)]):
    actual = ok[name].to_numpy()
    ok[name + '_predicted'] = calc
    metrics[name] = {group: float(100*np.mean(np.abs(calc[mask]/actual[mask]-1)))
                    for group, mask in [('base_mape_pct', fit_rows), ('extra_mape_pct', ~fit_rows)]}
    for mask, label, marker in [(fit_rows, 'Основные', 'o'), (~fit_rows, 'Дополнительные', 'x')]:
        ax.scatter(actual[mask], calc[mask], label=label, marker=marker, s=20)
    limits = [min(actual.min(), calc.min()), max(actual.max(), calc.max())]
    ax.plot(limits, limits, 'k--', linewidth=1)
    ax.set(xscale='log', yscale='log', xlabel=f'Измерение, {unit}', ylabel=f'Формула, {unit}')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / 'figures' / 'comparison.png', dpi=180)
plt.close(fig)

# Кривые по B для каждого размера S: точками показаны реальные замеры.
for name, unit, calc in [('latency_s','s',lambda s,b: latency(s,b,theta)),
                         ('memory_bytes','bytes',memory),
                         ('energy_j','J',lambda s,b: energy(s,b,theta_energy))]:
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, size in zip(axes.flat, sorted(ok.S.unique())):
        part = ok[ok.S == size].sort_values('B')
        bb = np.arange(1, 257)
        ax.plot(bb, calc(size, bb), label='Формула')
        for validation, marker, label in [(0,'o','Основные'),(1,'x','Дополнительные')]:
            pts = part[part.is_validation == validation]
            ax.scatter(pts.B, pts[name], s=14, marker=marker, label=label)
        ax.set(xscale='log', yscale='log', title=f'S = {size}', xlabel='B', ylabel=unit)
    axes.flat[-1].axis('off')
    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc='center')
    fig.tight_layout()
    fig.savefig(out / 'figures' / f'{name}.png', dpi=160)
    plt.close(fig)

counts = pd.read_csv(out / 'flops.csv')
mac = 17712 * counts.S**2 + 313344
metrics['flops'] = dict(profiler_mac_max_abs_error=float(np.max(np.abs(counts.flops_profiled-mac))),
                       convention='MAC=2; ReLU/MaxPool comparison=1; GAP and Linear bias included')
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(counts.S, mac, label='Conv/Linear: формула')
ax.plot(counts.S, flops(counts.S,1), '--', label='Все операции: формула')
ax.scatter(counts.S, counts.flops_profiled, marker='x', label='Profiler: Conv/Linear')
ax.set(xlabel='S', ylabel='Операций на проход, B = 1')
ax.legend(); fig.tight_layout()
fig.savefig(out / 'figures' / 'flops.png', dpi=180)
plt.close(fig)
ok['flops_predicted'] = flops(s,b)
ok['memory_formula'] = memory(s,b)
ok['oom_formula'] = ok.memory_formula > ok.memory_budget_bytes
ok.to_csv(out / 'predictions.csv', index=False)
metrics['grid'] = dict(total=len(df), base=int(fit_rows.sum()), extra=int((~fit_rows).sum()),
                       actual_oom=int(df.status.eq('OOM').sum()),
                       predicted_oom=int((memory(df.S,df.B)>df.memory_budget_bytes).sum()))
(out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(metrics, ensure_ascii=False, indent=2))
