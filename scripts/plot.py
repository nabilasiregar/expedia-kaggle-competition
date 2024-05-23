import matplotlib.pyplot as plt
import xgboost as xgb


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 24,
    "font.weight": 'bold',
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
    "figure.figsize": (15, 10),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.format": "png",
    # "savefig.transparent": True,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "0.8",
    "image.cmap": "Blues",
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "text.usetex": True, "mathtext.fontset": "cm",
    "pgf.preamble": r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage{cmbright}"
})


model = xgb.Booster()
model.load_model('models/final_model.json')

xgb.plot_importance(model, max_num_features=15, height=0.4)
plt.savefig('../../Feature_importance.png', bbox_inches='tight')
