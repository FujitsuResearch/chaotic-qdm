import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../source/utils/')
import plot_utils as putils

if __name__ == "__main__":
    # Check for command line arguments
    parser = argparse.ArgumentParser()

    # Full directory
    parser.add_argument('--full_dir', type=str, default='../results_2026/20260112_train_qm9_mol_8_2/qm9_qubits_7_dat_4236_npj_3')
    parser.add_argument('--full_base', type=str, default='qm9_atoms_8_rings_2_Ising_nearest_3_full_0.02_rxycz_qubits_7_3_steps_20_wass_lays_10_in_product_dat_2000_4236_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')

    # Latent directory
    parser.add_argument('--lat_dir', type=str, default='../results_2026/20251224_train_qm9_mol_8_2/qm9_qubits_7_dat_4236_npj_2')
    parser.add_argument('--lat_base', type=str, default='qm9_atoms_8_rings_2_Ising_nearest_2_full_0.02_qae_latent_4_lays_20_epoch_2000_lr_0.001_rxycz_qubits_7_2_steps_20_wass_lays_10_in_product_dat_2000_4236_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')

    parser.add_argument('--ymin', type=float, default=1e-1)
    parser.add_argument('--ymax', type=float, default=1)
    
    args = parser.parse_args()
    full_dir, full_base = args.full_dir, args.full_base
    lat_dir, lat_base = args.lat_dir, args.lat_base
        
    # Plot dist
    lw = 3
    mkz = 9
    putils.setPlot(fontsize=24, labelsize=24, lw=lw)
    
    fig, axs = plt.subplots(1, 3, figsize=(21,8), squeeze=False, sharey=False)
    axs = axs.ravel()
    ax, bx, cx = axs[0], axs[1], axs[2]
    ax.set_title('CTED', fontsize=28)
    bx.set_title('RTED', fontsize=28)
    cx.set_title('RUCD', fontsize=28)
    #putils.set_axes_facecolor(axs)

    full_tmps = ['Ising_nearest_3_full_0.02', 'Ising_nearest_3_trotter_0.02', 'random_10.0']
    lat_tmps = ['Ising_nearest_2_full_0.02', 'Ising_nearest_2_trotter_0.02', 'random_10.0']
    methods = ['CTED', 'RTED', 'RUCD']
    colors = [putils.RED_m, putils.GREEN_m, putils.BLUE_m, putils.BROWN]

    for i in range(3):
        ax = axs[i]
        #ax.set_title(f'{methods[i]}', fontsize=28)
        col = colors[i]
        method = methods[i]
        base1 = full_base.replace('Ising_nearest_3_full_0.02', full_tmps[i])
        base2 = lat_base.replace('Ising_nearest_2_full_0.02', lat_tmps[i])

        real_wass_1_ls, real_wass_2_ls, test_wass_1_ls, test_wass_2_ls = [], [], [], []
        for seed in range(50):
            real_file = os.path.join(full_dir, f'res/{base1}_seed_{seed}_DIST.npy')

            test_file_1 = os.path.join(full_dir, f'res/{base1}_seed_{seed}_test_DIST.npy')
            test_file_2 = os.path.join(lat_dir, f'res/{base2}_seed_{seed}_test_DIST.npy')
            
            if os.path.isfile(real_file):
                real_wass_1 = np.load(real_file)[1]
                real_wass_1_ls.append(real_wass_1)

            if os.path.isfile(test_file_1):
                test_wass_1 = np.load(test_file_1)[1]
                test_wass_1_ls.append(test_wass_1)
            
            if os.path.isfile(test_file_2):
                test_wass_2 = np.load(test_file_2)[1]
                test_wass_2_ls.append(test_wass_2)

        # if len(real_wass_1_ls) > 0:
        #     real_wass_1_ls = np.array(real_wass_1_ls)
        #     avg_real_wass_1, svd_real_wass_1 = np.mean(real_wass_1_ls, axis=0), np.std(real_wass_1_ls, axis=0)
        #     ax.plot(avg_real_wass_1, '--', mfc='white', markersize=mkz, c=putils.RED_m, lw=lw, label=rf'Forward')
        #     ax.fill_between(np.arange(avg_real_wass_1.shape[0]), avg_real_wass_1 - svd_real_wass_1, avg_real_wass_1 + svd_real_wass_1, color='gray', alpha=0.2)

        if len(test_wass_1_ls) > 0:
            test_wass_1_ls = np.array(test_wass_1_ls)
            avg_test_wass_1, svd_test_wass_1 = np.mean(test_wass_1_ls, axis=0), np.std(test_wass_1_ls, axis=0)
            ax.plot(avg_test_wass_1, '--', mfc='white', markersize=mkz, c=col, lw=lw, label=rf'{method}-Full')
            ax.fill_between(np.arange(avg_test_wass_1.shape[0]), avg_test_wass_1 - svd_test_wass_1, avg_test_wass_1 + svd_test_wass_1, color=colors[-1], alpha=0.2)

        if len(test_wass_2_ls) > 0:
            test_wass_2_ls = np.array(test_wass_2_ls)
            avg_test_wass_2, svd_test_wass_2 = np.mean(test_wass_2_ls, axis=0), np.std(test_wass_2_ls, axis=0)

            ax.plot(avg_test_wass_2, 'o-', mfc='white', markersize=mkz, c=col, lw=lw, label=rf'{method}-Latent')
            ax.fill_between(np.arange(avg_test_wass_2.shape[0]), avg_test_wass_2 - svd_test_wass_2, avg_test_wass_2 + svd_test_wass_2, color=col, alpha=0.2)


    #ax.legend(fontsize=20, framealpha=0, ncol=2, columnspacing=0.4, loc='upper left', bbox_to_anchor=(-0.1, 1.35))
    #ax.set_xlim([0, 48])
    #ax.legend(loc='lower right', fontsize=20, framealpha=0, ncol=2, columnspacing=0.4)
    #ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    putils.set_axes_tick1(axs, xlabel='Step $k$', legend=True, tick_minor=True, top_right_spine=True, w=3, tick_length_unit=5)
    #putils.set_axes_tick1([ax2, bx2, cx2], tick_minor=True, top_right_spine=True, w=2, tick_length_unit=5, legend=False)
    
    for ax in axs:
        ax.legend(loc='lower right', fontsize=22, framealpha=0.8)
        #ax.set_yscale('log')
        ax.set_ylim([1e-3, None])
        ax.set_ylabel(r'Wass. Dist.')
    
    plt.tight_layout()
    
    for tmp in zip([full_dir, lat_dir], [full_base, lat_base]):
        folder, basename = tmp
        if os.path.exists(folder):
            fig_folder = os.path.join(folder, 'figs_diff')
            os.makedirs(fig_folder, exist_ok=True)
            fig_file = os.path.join(fig_folder, f'compare_full_latent_{basename}')

            for ftype in ['png', 'svg', 'pdf']:
                plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()