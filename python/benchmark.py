import matplotlib.pyplot as plt
import numpy as np

# Données des résolutions avec leur nombre de pixels
resolution_info = {
    '144p': '144p\n36 864 px',
    '240p': '240p\n102 240 px', 
    '360p': '360p\n230 400 px',
    '480p': '480p\n409 920 px',
    '720p': '720p\n921 600 px',
    '1080p': '1080p\n2 073 600 px'
}

resolutions = list(resolution_info.keys())
pixel_labels = [resolution_info[res] for res in resolutions]  # Définition correcte de pixel_labels

# Données des temps d'exécution
cpp_times = [
    [0.0152274, 0.0219499, 0.0168805, 0.0145093, 0.0130096, 0.0173652],
    [0.0391694, 0.0524609, 0.0460058, 0.0378348, 0.0239367, 0.0286055],
    [0.0963637, 0.158475, 0.0976397, 0.0725043, 0.0545947, 0.063702],
    [0.167537, 0.361794, 0.19602, 0.174795, 0.101341, 0.114408],
    [0.553915, 0.86573, 0.603233, 0.434434, 0.25148, 0.280451],
    [2.499658, 2.7596, 2.12577, 2.54134, 0.675587, 0.74818]
]

python_times = [
    [1.114808, 1.557223, 1.165483, 1.176984, 0.942423, 1.057431],
    [3.678505, 3.959726, 1.598547, 1.609259, 1.12193, 1.267098],
    [5.548335, 6.532212, 4.341291, 4.295977, 2.611824, 2.884495],
    [8.319435, 13.834774, 8.868423, 8.764364, 4.812249, 5.048876],
    [22.663801, 43.271901, 25.071915, 25.653564, 11.77943, 11.6203],
    [86.139482, 129.907835, 75.732844, 79.558203, 29.497051, 27.147964]
]

# Calcul des moyennes
cpp_means = [np.mean(times) for times in cpp_times]
python_means = [np.mean(times) for times in python_times]
speed_factors = [py/cpp for cpp, py in zip(cpp_means, python_means)]

# Création du graphique
plt.figure(figsize=(16, 9))

# Configuration du style
plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
cpp_color = '#1a5276'
python_color = '#e67e22'
factor_color = '#27ae60'

# Nuage de points
for i, res in enumerate(resolutions):
    plt.scatter([res]*len(cpp_times[i]), cpp_times[i], color=cpp_color, alpha=0.6, s=90, 
                label='Parallèle CUDA' if i==0 else "")
    plt.scatter([res]*len(python_times[i]), python_times[i], color=python_color, alpha=0.6, s=90,
                label='Séquentielle' if i==0 else "")

# Lignes de moyennes
plt.plot(resolutions, cpp_means, color=cpp_color, marker='o', markersize=12, linewidth=3.5, 
         label='Moyenne Parallèle CUDA')
plt.plot(resolutions, python_means, color=python_color, marker='o', markersize=12, linewidth=3.5,
         label='Moyenne Séquentielle')

# Annotations des facteurs
for i, (cpp_val, py_val) in enumerate(zip(cpp_means, python_means)):
    mid_y = (cpp_val * py_val)**0.5
    factor_text = f'×{speed_factors[i]:.0f}' if speed_factors[i] > 9 else f'×{speed_factors[i]:.1f}'
    
    plt.text(i, mid_y, factor_text, 
             ha='center', va='center',
             fontsize=14, weight='bold', color=factor_color,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor=factor_color, boxstyle='round,pad=0.3'))

# Configuration des axes
ax = plt.gca()
ax.set_yscale('log')

# Définition des ticks avec les labels combinés
ax.set_xticks(np.arange(len(resolutions)))
ax.set_xticklabels(pixel_labels, fontsize=10, linespacing=1.5)

plt.ylabel('Temps d\'exécution (s) - Échelle logarithmique', fontsize=12)
plt.xlabel('Résolution', fontsize=12)
plt.title('Benchmark: Parallèle CUDA vs Séquentielle', fontsize=16, pad=20)

# Légende
handles, labels = ax.get_legend_handles_labels()
order = [0, 2, 1, 3]
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], 
           fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('benchmark_with_pixels.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé sous 'benchmark_with_pixels.png'")