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

    parser.add_argument('--dir1', type=str, default='../results/20251021_train_QDM_multi_cluster/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base1', type=str, default='multi_cluster_random_10.0_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')
    
    parser.add_argument('--dir2', type=str, default='../results/20251021_train_QDM_multi_cluster/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base2', type=str, default='multi_cluster_Ising_nearest_2_full_0.02_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')
    
    parser.add_argument('--dir3', type=str, default='../results/20251021_train_QDM_multi_cluster/multi_cluster_qubits_4_dat_3000_npj_2')
    parser.add_argument('--base3', type=str, default='multi_cluster_Ising_nearest_2_trotter_0.02_rxycz_qubits_4_2_steps_20_wass_lays_10_in_product_dat_1000_3000_epoch_1001_100_lr_0.001_init_1.0_vd_0.0')

    parser.add_argument('--ymin', type=float, default=1e-3)
    parser.add_argument('--ymax', type=float, default=1e-3)
    
    args = parser.parse_args()

    dir1, base1 = args.dir1, args.base1
    dir2, base2 = args.dir2, args.base2
    dir3, base3 = args.dir3, args.base3

    real_wass_1_ls, real_wass_2_ls, real_wass_3_ls, test_wass_1_ls, test_wass_2_ls, test_wass_3_ls = [], [], [], [], [], []
    for seed in range(10):
        real_file_1 = os.path.join(dir1, f'res/{base1}_seed_{seed}_DIST.npy')
        test_file_1 = os.path.join(dir1, f'res/{base1}_seed_{seed}_test_DIST.npy')
        real_file_2 = os.path.join(dir2, f'res/{base2}_seed_{seed}_DIST.npy')
        test_file_2 = os.path.join(dir2, f'res/{base2}_seed_{seed}_test_DIST.npy')
        real_file_3 = os.path.join(dir3, f'res/{base3}_seed_{seed}_DIST.npy')
        test_file_3 = os.path.join(dir3, f'res/{base3}_seed_{seed}_test_DIST.npy')

        if os.path.isfile(real_file_1) and os.path.isfile(test_file_1):
            real_wass_1 = np.load(real_file_1)[1]
            test_wass_1 = np.load(test_file_1)[1]
            real_wass_1_ls.append(real_wass_1)
            test_wass_1_ls.append(test_wass_1)
        
        if os.path.isfile(real_file_2) and os.path.isfile(test_file_2):
            real_wass_2 = np.load(real_file_2)[1]
            test_wass_2 = np.load(test_file_2)[1]
            real_wass_2_ls.append(real_wass_2)
            test_wass_2_ls.append(test_wass_2)

        if os.path.isfile(real_file_3) and os.path.isfile(test_file_3):
            real_wass_3 = np.load(real_file_3)[1]
            test_wass_3 = np.load(test_file_3)[1]
            real_wass_3_ls.append(real_wass_3)
            test_wass_3_ls.append(test_wass_3)

    real_wass_1_ls = np.array(real_wass_1_ls)
    test_wass_1_ls = np.array(test_wass_1_ls)
    real_wass_2_ls = np.array(real_wass_2_ls)
    test_wass_2_ls = np.array(test_wass_2_ls)
    real_wass_3_ls = np.array(real_wass_3_ls)
    test_wass_3_ls = np.array(test_wass_3_ls)
    
    avg_real_wass_1, svd_real_wass_1 = np.mean(real_wass_1_ls, axis=0), np.std(real_wass_1_ls, axis=0)
    avg_test_wass_1, svd_test_wass_1 = np.mean(test_wass_1_ls, axis=0), np.std(test_wass_1_ls, axis=0)
    avg_real_wass_2, svd_real_wass_2 = np.mean(real_wass_2_ls, axis=0), np.std(real_wass_2_ls, axis=0)
    avg_test_wass_2, svd_test_wass_2 = np.mean(test_wass_2_ls, axis=0), np.std(test_wass_2_ls, axis=0)
    avg_real_wass_3, svd_real_wass_3 = np.mean(real_wass_3_ls, axis=0), np.std(real_wass_3_ls, axis=0)
    avg_test_wass_3, svd_test_wass_3 = np.mean(test_wass_3_ls, axis=0), np.std(test_wass_3_ls, axis=0)
    
    print(test_wass_1_ls.shape, avg_real_wass_1.shape)

    # Plot dist
    lw = 3
    mkz = 9
    putils.setPlot(fontsize=26, labelsize=24, lw=lw)
    
    fig, axs = plt.subplots(1, 3, figsize=(16,7), squeeze=False, sharey=True)
    axs = axs.ravel()
    ax, bx, cx = axs[0], axs[1], axs[2]
    #putils.set_axes_facecolor(axs)

    xs = np.arange(avg_real_wass_1.shape[0])
    
    if len(avg_real_wass_2) > 0 and len(avg_test_wass_2) > 0:
        ax.plot(avg_test_wass_2, 'o-', mfc='white', markersize=mkz, c=putils.RED_m, lw=lw, label=r'CTED-backward')
        ax.fill_between(np.arange(avg_test_wass_2.shape[0]), avg_test_wass_2 - svd_test_wass_2, avg_test_wass_2 + svd_test_wass_2, color=putils.RED_m, alpha=0.2)
        
        ax.plot(avg_real_wass_2, '--', mfc='white', markersize=mkz, c=putils.RED_m, lw=lw, label=r'CTED-forward')
        ax.fill_between(np.arange(avg_real_wass_2.shape[0]), avg_real_wass_2 - svd_real_wass_2, avg_real_wass_2 + svd_real_wass_2, color='gray', alpha=0.2)
    
    if len(avg_real_wass_3) > 0 and len(avg_test_wass_3) > 0:
        bx.plot(avg_test_wass_3, 'o-', mfc='white', markersize=mkz, c=putils.GREEN_m, lw=lw, label=r'RTED-backward')
        bx.fill_between(np.arange(avg_test_wass_3.shape[0]), avg_test_wass_3 - svd_test_wass_3, avg_test_wass_3 + svd_test_wass_3, color=putils.BLUE_m, alpha=0.2)

        bx.plot(avg_real_wass_3, '--', mfc='white', markersize=mkz, c=putils.GREEN_m, lw=lw, label=r'RTED-forward')
        bx.fill_between(np.arange(avg_real_wass_3.shape[0]), avg_real_wass_3 - svd_real_wass_3, avg_real_wass_3 + svd_real_wass_3, color='gray', alpha=0.2)

    if len(avg_real_wass_1) > 0 and len(avg_test_wass_1) > 0:
        cx.plot(avg_test_wass_1, 'o-', mfc='white', markersize=mkz, c=putils.BLUE_m, lw=lw, label=r'RUCD-backward')
        cx.fill_between(np.arange(avg_test_wass_1.shape[0]), avg_test_wass_1 - svd_test_wass_1, avg_test_wass_1 + svd_test_wass_1, color=putils.BLUE_m, alpha=0.2)

        cx.plot(avg_real_wass_1, '--', mfc='white', markersize=mkz, c=putils.BLUE_m, lw=lw, label=r'RUCD-forward')
        cx.fill_between(np.arange(avg_real_wass_1.shape[0]), avg_real_wass_1 - svd_real_wass_1, avg_real_wass_1 + svd_real_wass_1, color='gray', alpha=0.2)
    


    #ax.legend(fontsize=20, framealpha=0, ncol=2, columnspacing=0.4, loc='upper left', bbox_to_anchor=(-0.1, 1.35))
    ax.set_yscale('log')
    ax.set_ylim([1e-3, None])
    ax.set_ylabel(r'Dist. $\mathcal{D}_{\text{Wass}}(\tilde{\mathcal{S}}^\prime_k,\mathcal{S}^\prime_0)$')
    y_min, y_max = args.ymin, args.ymax
    if y_min < y_max:
        ax.set_ylim([y_min, y_max])
    #ax.set_xlim([0, 48])
    #ax.legend(loc='lower right', fontsize=20, framealpha=0, ncol=2, columnspacing=0.4)
    #ax.tick_params(direction='in', length=10, width=3, top='on', right='on', labelsize=30)
    putils.set_axes_tick1(axs, xlabel='Step $k$', legend=True, tick_minor=True, top_right_spine=True, w=3, tick_length_unit=5)
    
    for ax in axs:
        ax.legend(loc='upper left', fontsize=20, framealpha=0.8)
    
    plt.tight_layout()
    
    for tmp in zip([dir1], [base1]):
        folder, basename = tmp
        if os.path.exists(folder):
            fig_folder = os.path.join(folder, 'figs_diff')
            os.makedirs(fig_folder, exist_ok=True)
            fig_file = os.path.join(fig_folder, f'compare_eval_{basename}')

            for ftype in ['png', 'svg', 'pdf']:
                plt.savefig('{}.{}'.format(fig_file, ftype), bbox_inches = 'tight', dpi=300)
    plt.show()
    plt.clf()
    plt.close()