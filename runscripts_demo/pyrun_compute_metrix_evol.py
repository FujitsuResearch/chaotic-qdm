#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np 
import os 
import multiprocessing as mp
import argparse


# Network params
class NParams():
    def __init__(self, n_qubits, n_diff_steps, scramb, type_evol, n_pj_qubits, delta_t, rand_ancilla, J, bz, W):
        self.n_qubits = n_qubits
        self.n_diff_steps = n_diff_steps
        self.scramb = scramb
        self.type_evol = type_evol
        self.n_pj_qubits = n_pj_qubits
        self.delta_t = delta_t
        self.rand_ancilla = rand_ancilla
        # For Ising_random_all
        self.J = J
        self.bz = bz
        self.W = W

# Data params
class DParams():
    def __init__(self, dat_name, n_dat):
        self.dat_name = dat_name
        self.n_dat = n_dat
        
def execute_job(bin, nparams, dparams, save_dir, rseed, k_moment):
    print(f'Start process with rseed={rseed}')
    cmd = f'python {bin} \
            --n_qubits {nparams.n_qubits} \
            --n_diff_steps {nparams.n_diff_steps} \
            --scramb {nparams.scramb} \
            --type_evol {nparams.type_evol} \
            --n_pj_qubits {nparams.n_pj_qubits} \
            --delta_t {nparams.delta_t} \
            --rand_ancilla {nparams.rand_ancilla} \
            --J {nparams.J} \
            --bz {nparams.bz} \
            --W {nparams.W} \
            --dat_name {dparams.dat_name} \
            --n_dat {dparams.n_dat} \
            --rseed {rseed} \
            --k_moment {k_moment} \
            --save_dir {save_dir}'
    os.system(cmd)
    print(f'Finish process with rseed={rseed}')

    
if __name__ == '__main__':
    #mp.set_start_method("spawn")

    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='../results/del_test2')
    parser.add_argument('--bin', type=str, default='../source/ensemble_evol.py')

    # For diffusion setting
    parser.add_argument('--scramb', type=str, default='Ising_nearest', help='Type of scrambling: random, Ising_nearest, Ising_random_all')
    parser.add_argument('--type_evol', type=str, default='full', help='Type of evolution: full or trotter')
    parser.add_argument('--n_pj_qubits', type=str, default='4', help='number of projective qubits')
    parser.add_argument('--delta_t', type=str, default='0.01', help='time interval for evolution')
    parser.add_argument('--rand_ancilla', type=int, default=1, help='random basis for ancilla or not, 1 for True, 0 for False')
    parser.add_argument('--n_diff_steps', type=int, default=10, help='number steps for diffusion')
    
    # For Ising_random_all
    parser.add_argument('--J', type=str, default='1.0', help='J coefficient for Ising_random_all')
    parser.add_argument('--bz', type=str, default='0.0', help='Bz coefficient for Ising_random_all')
    parser.add_argument('--W', type=str, default='1.0', help='W coefficient for Ising_random_all')


    # For data
    parser.add_argument('--dat_name', type=str, default='multi_cluster', help='name of the data: cluster0, line, circle, tfim, multi_cluster')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of data qubits')
    parser.add_argument('--n_dat', type=int, default=100, help='Number of data instances')

    # For system
    parser.add_argument('--rseed', type=str, default='0', help='Random seed')

    # For moment order
    parser.add_argument('--k_moment', type=int, default=2, help='Moment order k for distance calculation')

    args = parser.parse_args()

    save_dir, n_diff_steps = args.save_dir, args.n_diff_steps
    dat_name, n_dat, n_qubits, rseed = args.dat_name, args.n_dat, args.n_qubits, args.rseed
    scramb, type_evol, n_pj_qubits, delta_t, rand_ancilla = args.scramb, args.type_evol, args.n_pj_qubits, args.delta_t, args.rand_ancilla
    b, J, W = args.bz, args.J, args.W

    k_moment = args.k_moment
    scramb_ls = [str(x) for x in args.scramb.split(',')]
    n_pj_ls = [str(x) for x in args.n_pj_qubits.split(',')]
    type_evol_ls = [str(x) for x in args.type_evol.split(',')]
    delta_ls = [str(x) for x in args.delta_t.split(',')]
    
    dat_names = [str(x) for x in args.dat_name.split(',')]
    rseeds = [int(x) for x in args.rseed.split(',')]
    
    # for Ising_random_all
    J = float(args.J)
    bz_ls = [float(x) for x in args.bz.split(',')]
    W_ls = [float(x) for x in args.W.split(',')]

    args_list = []
    for dat_name in dat_names:
        dparams = DParams(dat_name, n_dat)
        for scramb in scramb_ls:
            for n_pj_qubits in n_pj_ls:
                for type_evol in type_evol_ls:
                    if scramb == 'random' and type_evol == 'trotter':
                        print(f"Skip {scramb}-{type_evol}")
                        continue
                    for delta_t in delta_ls:
                        for bz in bz_ls:
                            for W in W_ls:
                                # Create the network parameters
                                nparams = NParams(
                                    n_qubits=n_qubits,
                                    n_diff_steps=n_diff_steps,
                                    scramb=scramb,
                                    type_evol=type_evol,
                                    n_pj_qubits=n_pj_qubits,
                                    delta_t=delta_t,
                                    rand_ancilla=args.rand_ancilla,
                                    J=J,
                                    bz=bz,
                                    W=W
                                )

                                for rseed in rseeds:
                                    args_list.append((
                                        args.bin,
                                        nparams,
                                        dparams,
                                        save_dir,
                                        rseed,
                                        k_moment
                                    ))
    n_workers = min(mp.cpu_count(), len(args_list))

    with mp.Pool(processes=n_workers) as pool:
        # starmap will call execute_job(*args_tuple) for each entry in job_args
        pool.starmap(execute_job, args_list, chunksize=1)