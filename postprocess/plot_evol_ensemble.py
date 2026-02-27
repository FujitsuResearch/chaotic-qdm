import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path


def _add_utils_to_path() -> None:
    """Ensure the local utils module is importable regardless of CWD."""
    script_dir = Path(__file__).resolve().parent
    utils_dir = script_dir.parent / 'source' / 'utils'
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))


_add_utils_to_path()
import plot_utils as putils


if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../results_2026/20251217_evol_multi_cluster')
    parser.add_argument('--dat_name', type=str, default='multi_cluster')
    parser.add_argument('--n_qubits', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--dat_size', type=int, default=1000)
    parser.add_argument('--keystr', type=str, default='HS_delta')
    args = parser.parse_args()

    folder, n_qubits, steps, dat_size, dat_name = args.folder, args.n_qubits, args.steps, args.dat_size, args.dat_name
    keystr = args.keystr

    if keystr == 'HS_delta':
        labelstr = r'$\Delta^{(m)}_2$'
    elif keystr == 'Wass':
        labelstr = r'Wass Dist.'       
    data_types = ['target', 'haar', 'haar_product']
    npjs = [2, 4, 6, 8, 10]
    k_moments = [1, 2, 3, 4]

    lw = 3
    mkz = 8

    fig_dir = os.path.join(folder, 'figs_dist')
    os.makedirs(fig_dir, exist_ok=True)
    
    for data_type in data_types:
      putils.setPlot(fontsize=30, labelsize=30, lw=lw)
      fig, axs = plt.subplots(2, len(k_moments), figsize=(8*len(k_moments), 16), squeeze=False)
      basename = f'{dat_name}_Ising_nearest_NPJ_SCRAMB_0.05_rdanc_1_qubits_{n_qubits}_steps_{steps}_dat_{dat_size}'

      for scramb in ['full', 'trotter']:
        for k in k_moments:
            ax = axs[0 if scramb == 'full' else 1, k - 1]
            axs[0, k-1].set_xlim(0, 200)
        
            if data_type == 'target':
                bx = ax.inset_axes([0.2, 0.12, 0.7, 0.47])
                bx.set_xlim(0, 50)
            else:
                bx = None
            #axs[1, k-1].set_xlim(0, 50)
            #ax.set_title(f'$m={k}$', fontsize=24, pad=-40)
            color = putils.modern[k - 1]
            for npj in npjs:
                sub_folder = os.path.join(folder, f'{dat_name}_qubits_{n_qubits}_dat_{dat_size}_npj_{npj}')
                dists = []
                for seed in range(20):
                  metric_base = basename.replace('SCRAMB', scramb)
                  metric_base = metric_base.replace('NPJ', str(npj)) + f'_seed_{seed}_dist_k={k}_{data_type}'
                  dist_metric_file = os.path.join(sub_folder, 'dist', f'{metric_base}.npz')
                  
                  if os.path.exists(dist_metric_file):
                      loaded = np.load(dist_metric_file)
                      metrics = {key: loaded[key] for key in loaded}
                      dists.append(metrics[keystr])
                if len(dists) == 0:
                    continue
                print(f'Plotting {data_type}, {scramb}, m={k}, npj={npj}, n_runs={len(dists)}')
                dists = np.array(dists).reshape(len(dists), -1).mean(axis=0)
                #ax.set_title(f'$m={k}$', fontsize=12)
                ax.plot(dists, '-',  mfc='white', markersize=9, c=color, lw=lw, alpha = npjs.index(npj)/len(npjs) + 0.3)
                if bx is not None:
                    bx.plot(dists, 'o-',  mfc='white', markersize=9, c=color, lw=lw, alpha = npjs.index(npj)/len(npjs) + 0.3)
                    putils.set_axes_tick1([bx], legend=False, tick_minor=False, top_right_spine=True, w=3, tick_length_unit=5)

      putils.set_axes_tick1(axs.ravel(), xlabel='Steps', ylabel=labelstr, legend=False, tick_minor=False, top_right_spine=True, w=3, tick_length_unit=5)
      fig_file = os.path.join(fig_dir, f'evol_dist_{data_type}_{keystr}_{basename}')
      plt.tight_layout()
      for ftype in ['pdf', 'png', 'svg']:
          plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
      plt.show()
      plt.clf()
      plt.close()
