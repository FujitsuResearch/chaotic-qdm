import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, '../source/utils')
import plot_utils as putils

if __name__ == "__main__":
    # Check for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir1', type=str, default='../results_2026/20260207_train_QDM_multi_cluster_noise/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base1', type=str, default='multi_cluster_random_10.0_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')
    
    parser.add_argument('--dir2', type=str, default='../results_2026/20260224_train_QDM_multi_cluster_noise/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base2', type=str, default='multi_cluster_Ising_nearest_2_full_0.02_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')
    
    parser.add_argument('--dir3', type=str, default='../results_2026/20260224_train_QDM_multi_cluster_noise/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base3', type=str, default='multi_cluster_Ising_nearest_2_trotter_0.02_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')

    parser.add_argument('--ymin', type=float, default=1e-3)
    parser.add_argument('--ymax', type=float, default=1e-3)
    
    args = parser.parse_args()

    dir1, org_base1 = args.dir1, args.base1
    dir2, org_base2 = args.dir2, args.base2
    dir3, org_base3 = args.dir3, args.base3

    # Plot dist
    lw = 3
    mkz = 12
    putils.setPlot(fontsize=24, labelsize=24, lw=lw)
    
    fig, axs = plt.subplots(1, 3, figsize=(21,8), squeeze=False, sharey=False)
    axs = axs.ravel()
    ax, bx, cx = axs[0], axs[1], axs[2]
    ax.set_title('CTED', fontsize=28)
    bx.set_title('RTED', fontsize=28)
    cx.set_title('RUCD', fontsize=28)
    #putils.set_axes_facecolor(axs)

    for noise in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        base1 = org_base1.replace('random_10.0_rxycz', f'random_10.0_noise_1_{noise}_rxycz')
        base1b = org_base1.replace('random_10.0_rxycz', f'random_10.0_noise_2_{noise}_rxycz')
        base2 = org_base2.replace('full_0.02_rxycz', f'full_0.02_noise_4_{noise}_rxycz')
        base3 = org_base3.replace('trotter_0.02_rxycz', f'trotter_0.02_noise_4_{noise}_rxycz')
        
        test_wass_1_ls, test_wass_1b_ls, test_wass_2_ls, test_wass_3_ls = [], [], [], []
        for seed in range(50):
            test_file_1 = os.path.join(dir1, f'res/{base1}_seed_{seed}_test_DIST.npy')
            test_file_1b = os.path.join(dir1, f'res/{base1b}_seed_{seed}_test_DIST.npy')
            test_file_2 = os.path.join(dir2, f'res/{base2}_seed_{seed}_test_DIST.npy')
            test_file_3 = os.path.join(dir3, f'res/{base3}_seed_{seed}_test_DIST.npy')
            
    
            if os.path.isfile(test_file_1):
                test_wass_1 = np.load(test_file_1)[1]
                test_wass_1_ls.append(test_wass_1[1])
            
            if os.path.isfile(test_file_1b):
                test_wass_1b = np.load(test_file_1b)[1]
                test_wass_1b_ls.append(test_wass_1b[1])
            
            if os.path.isfile(test_file_2):
                test_wass_2 = np.load(test_file_2)[1]
                test_wass_2_ls.append(test_wass_2[1])

            if os.path.isfile(test_file_3):
                test_wass_3 = np.load(test_file_3)[1]
                test_wass_3_ls.append(test_wass_3[1])
        
        test_wass_1_ls = np.array(test_wass_1_ls)
        test_wass_1b_ls = np.array(test_wass_1b_ls)
        test_wass_2_ls = np.array(test_wass_2_ls)
        test_wass_3_ls = np.array(test_wass_3_ls)
        # plot scatter
        cx.scatter([noise]*len(test_wass_1_ls), test_wass_1_ls, marker='o', s=mkz**2, edgecolors='k', linewidths=0.5, alpha=0.8, facecolors=putils.BLUE_m)
        #ax.scatter([noise], np.mean(test_wass_1b_ls), marker='o', s=mkz**2, edgecolors='k', linewidths=0.5, alpha=0.8, facecolors=putils.BLUE_m)
        ax.scatter([noise]*len(test_wass_2_ls), test_wass_2_ls, marker='s', s=mkz**2, edgecolors='k', linewidths=0.5, alpha=0.8, facecolors=putils.RED_m)
        bx.scatter([noise]*len(test_wass_3_ls), test_wass_3_ls, marker='^', s=mkz**2, edgecolors='k', linewidths=0.5, alpha=0.8, facecolors=putils.GREEN_m)

    #ax.legend(fontsize=20, framealpha=0, ncol=2, columnspacing=0.4, loc='upper left', bbox_to_anchor=(-0.1, 1.35))
    #ax.set_xlim([0, 48])
    #ax.legend(loc='lower right', fontsize=20, framealpha=0, ncol=2, columnspacing=0.4)
    #ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    putils.set_axes_tick1(axs, xlabel='Noise level', legend=True, tick_minor=True, top_right_spine=True, w=3, tick_length_unit=5)
    
    for ax in axs:
        #ax.legend(loc='upper right', fontsize=18, framealpha=0.8, bbox_to_anchor=(1.0, 0.9))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-2, 1.2])
        ax.set_ylabel(r'Wass. Dist.')
    
    plt.tight_layout()
    
    for tmp in zip([dir1], [base1]):
        folder, basename = tmp
        if os.path.exists(folder):
            fig_folder = os.path.join(folder, 'figs_diff')
            os.makedirs(fig_folder, exist_ok=True)
            fig_file = os.path.join(fig_folder, f'compare_noise_{basename}')

            for ftype in ['png', 'svg', 'pdf']:
                plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()