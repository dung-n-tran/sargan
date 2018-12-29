from cycler import cycler
color_palette = ["#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#34495e", "#95a5a6", "#f1c40f", "#e67e22"]

def config_figures(mpl, color_palette):
    mpl.rcParams["axes.grid"] = False
    mpl.rcParams["grid.color"] = "#f5f5f5"
    mpl.rcParams["axes.facecolor"] = "#ededed"
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = False
    mpl.rcParams['axes.labelcolor'] = "grey"
    mpl.rcParams['xtick.color'] = 'grey'
    mpl.rcParams['ytick.color'] = 'grey'
    mpl.rcParams["figure.figsize"] = [4, 3]
    mpl.rcParams["axes.prop_cycle"] = cycler('color', color_palette)