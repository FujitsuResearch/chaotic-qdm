#!/usr/bin/env python
# # -*- coding: utf-8 -*-

import numpy as np 
import os 
import multiprocessing as mp
import argparse


# Network params
class NParams():
    def __init__(self, gen_circuit_type, n_qubits, n_ancilla, n_layers, n_diff_steps, scramb, type_evol, n_pj_qubits, \
            delta_t, rand_ancilla, J, bz, W, noise_level, noise_type):
        self.gen_circuit_type = gen_circuit_type
        self.n_qubits = n_qubits
        self.n_ancilla = n_ancilla
        self.n_layers = n_layers
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
        self.noise_level = noise_level
        self.noise_type = noise_type

# Data params
class DParams():
    def __init__(self, dat_name, input_type, n_train, n_test):
        self.dat_name = dat_name
        self.input_type = input_type
        self.n_train = n_train
        self.n_test = n_test

# Optimizer params
class OParams():
    def __init__(self, lr, mag, n_outer_epochs, batch_size, dist_type, vendi_lambda):        
        self.lr = lr
        self.mag = mag
        self.n_outer_epochs = n_outer_epochs
        self.batch_size = batch_size
        self.dist_type = dist_type
        self.vendi_lambda = vendi_lambda

class QAEParams():
    def __init__(self, use_qae, qae_latent, qae_layers, qae_epochs, qae_lr):
        self.use_qae = use_qae
        self.qae_latent = qae_latent
        self.qae_layers = qae_layers
        self.qae_epochs = qae_epochs
        self.qae_lr = qae_lr
        
def execute_job(bin, nparams, dparams, oparams, qaeparams, save_dir, load_params, rseed, plot_bloch, n_threads):
    print(f'Start process with rseed={rseed}')
    cmd = f'python {bin} \
            --n_qubits {nparams.n_qubits} \
            --n_ancilla {nparams.n_ancilla} \
            --n_layers {nparams.n_layers} \
            --n_diff_steps {nparams.n_diff_steps} \
            --scramb {nparams.scramb} \
            --type_evol {nparams.type_evol} \
            --n_pj_qubits {nparams.n_pj_qubits} \
            --delta_t {nparams.delta_t} \
            --rand_ancilla {nparams.rand_ancilla} \
            --J {nparams.J} \
            --bz {nparams.bz} \
            --W {nparams.W} \
            --gen_circuit_type {nparams.gen_circuit_type} \
            --noise_level {nparams.noise_level} \
            --noise_type {nparams.noise_type} \
            --dat_name {dparams.dat_name} \
            --input_type {dparams.input_type} \
            --n_train {dparams.n_train} \
            --n_test {dparams.n_test} \
            --n_outer_epochs {oparams.n_outer_epochs} \
            --batch_size {oparams.batch_size} \
            --dist_type {oparams.dist_type} \
            --lr {oparams.lr} \
            --mag {oparams.mag} \
            --vendi_lambda {oparams.vendi_lambda} \
            --use_qae {qaeparams.use_qae} \
            --qae_latent {qaeparams.qae_latent} \
            --qae_layers {qaeparams.qae_layers} \
            --qae_epochs {qaeparams.qae_epochs} \
            --qae_lr {qaeparams.qae_lr} \
            --rseed {rseed} \
            --bloch {plot_bloch} \
            --load_params {load_params} \
            --threads {n_threads} \
            --save_dir {save_dir}'
    os.system(cmd)
    print(f'Finish process with rseed={rseed}')

    
if __name__ == '__main__':
    #mp.set_start_method("spawn")

    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='../results/del_test2')
    parser.add_argument('--bin', type=str, default='../source/train_QDM.py')

    parser.add_argument('--load_params', type=str, default='params_file')

    # For diffusion setting
    parser.add_argument('--scramb', type=str, default='random', help='Type of scrambling: random, Ising_nearest, Ising_random_all')
    parser.add_argument('--type_evol', type=str, default='full')
    parser.add_argument('--n_pj_qubits', type=str, default=1, help='number of projective qubits')
    parser.add_argument('--delta_t', type=str, default=1.0, help='time interval for evolution')
    parser.add_argument('--rand_ancilla', type=int, default=1, help='random ancilla qubits, 1 for True, 0 for False')

    # For network
    parser.add_argument('--n_layers', type=str, default='10', help='number layers for generator')
    parser.add_argument('--n_diff_steps', type=int, default=40, help='number steps for diffusion')
    parser.add_argument('--J', type=float, default=1.0, help='J coefficient for Ising_random_all')
    parser.add_argument('--bz', type=str, default='0.0', help='Bz coefficient for Ising_random_all')
    parser.add_argument('--W', type=str, default='1.0', help='W coefficient for Ising_random_all')
      # For noise model
    parser.add_argument('--noise_level', type=str, default='0.0', help='Noise level for the model')
    parser.add_argument('--noise_type', type=int, default=0, help='Type of noise: 0 for no noise, 1 for single qubit depolarizing, 2 for two-qubit depolarizing, 3 for both, 4 for dephasing')

    # For data
    parser.add_argument('--dat_name', type=str, default='cluster0', help='name of the data: cluster0, line')
    parser.add_argument('--input_type', type=str, default='rand', help='type of the input')
    parser.add_argument('--n_qubits', type=int, default=1, help='Number of data qubits')
    parser.add_argument('--n_ancilla', type=int, default=1, help='Number of ancilla qubits')
    
    parser.add_argument('--n_train', type=int, default=100, help='Number of training data')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test data')

    # For epoch in training
    parser.add_argument('--n_outer_epochs', type=int, default=1000, help='Number of outer training epoch')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--dist_type', type=str, default='wass', help='Type of distance: wass or mmd')

    # For update matrix
    parser.add_argument('--lr', type=str, default='0.001', help='Learning rate for generator')
    parser.add_argument('--mag', type=str, default='1.0', help='Magnitude of initial parameters')
    parser.add_argument('--vendi_lambda', type=str, default='0.0', help='Vendi loss lambda')
    #parser.add_argument('--ep', type=float, default=0.01, help='Step size')

    # For QAE
    parser.add_argument('--use_qae', type=int, default=0, help='Use QAE or not, 1 for True, 0 for False')
    parser.add_argument('--qae_latent', type=int, default=2, help='Number of latent qubits for QAE')
    parser.add_argument('--qae_layers', type=int, default=5, help='Number of layers for QAE')
    parser.add_argument('--qae_epochs', type=int, default=1000, help='Number of training epochs for QAE')
    parser.add_argument('--qae_lr', type=float, default=0.001, help='Learning rate for QAE')

    # For gen circuit type
    parser.add_argument('--gen_circuit_type', type=str, default='rxycz', help='type of generator circuit')

    # For system
    parser.add_argument('--rseed', type=str, default='0', help='Random seed')
    parser.add_argument('--bloch', type=int, default=0, help='Plot bloch')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads')

    args = parser.parse_args()

    save_dir = args.save_dir
    n_train, n_test, n_outer_epochs, batch_size = args.n_train, args.n_test, args.n_outer_epochs, args.batch_size
    n_qubits, n_ancilla, rseed = args.n_qubits, args.n_ancilla, args.rseed
    load_params = args.load_params
    n_diff_steps = args.n_diff_steps
    plot_bloch, n_threads = args.bloch, args.threads

    gen_circuit_type = args.gen_circuit_type
    noise_level, noise_type = args.noise_level, args.noise_type

    use_qae, qae_latent, qae_layers, qae_epochs, qae_lr = args.use_qae, args.qae_latent, args.qae_layers, args.qae_epochs, args.qae_lr
    qae_params = QAEParams(use_qae, qae_latent, qae_layers, qae_epochs, qae_lr)

    input_type_ls = [str(x) for x in args.input_type.split(',')]
    scramb_ls = [str(x) for x in args.scramb.split(',')]
    n_pj_ls = [str(x) for x in args.n_pj_qubits.split(',')]
    type_evol_ls = [str(x) for x in args.type_evol.split(',')]
    delta_ls = [str(x) for x in args.delta_t.split(',')]
    noise_level_ls = [float(x) for x in args.noise_level.split(',')]

    n_layers_ls = [int(x) for x in args.n_layers.split(',')]
    dat_names = [str(x) for x in args.dat_name.split(',')]
    rseeds = [int(x) for x in args.rseed.split(',')]
    lrs = [float(x) for x in args.lr.split(',')]
    mags = [float(x) for x in args.mag.split(',')]
    vendi_lambdas = [float(x) for x in args.vendi_lambda.split(',')]
    dist_types = [str(x) for x in args.dist_type.split(',')]

    # for Ising_random_all
    J = float(args.J)
    bz_ls = [float(x) for x in args.bz.split(',')]
    W_ls = [float(x) for x in args.W.split(',')]

    args_list = []
    for dat_name in dat_names:
        for input_type in input_type_ls:
            dparams = DParams(dat_name, input_type, n_train, n_test)
            for n_layers in n_layers_ls:
                for scramb in scramb_ls:
                    for n_pj_qubits in n_pj_ls:
                        for type_evol in type_evol_ls:
                            if scramb == 'random' and type_evol == 'trotter':
                                print(f"Skip {scramb}-{type_evol}")
                                continue
                            for delta_t in delta_ls:
                                for bz in bz_ls:
                                    for W in W_ls:
                                        for noise_level in noise_level_ls:
                                            # Create the network parameters
                                            nparams = NParams(
                                                gen_circuit_type=gen_circuit_type,
                                                n_qubits=n_qubits,
                                                n_ancilla=n_ancilla,
                                                n_layers=n_layers,
                                                n_diff_steps=n_diff_steps,
                                                scramb=scramb,
                                                type_evol=type_evol,
                                                n_pj_qubits=n_pj_qubits,
                                                delta_t=delta_t,
                                                rand_ancilla=args.rand_ancilla,
                                                J=J,
                                                bz=bz,
                                                W=W,
                                                noise_level=noise_level,
                                                noise_type=noise_type
                                            )
                                            for lr in lrs:
                                                for mag in mags:
                                                    for vendi_lambda in vendi_lambdas:
                                                        for dist_type in dist_types:
                                                            oparams = OParams(
                                                                lr=lr,
                                                                mag=mag,
                                                                n_outer_epochs=n_outer_epochs,
                                                                batch_size=batch_size,
                                                                dist_type=dist_type,
                                                                vendi_lambda=vendi_lambda
                                                            )
                                                            for rseed in rseeds:
                                                                args_list.append((
                                                                    args.bin,
                                                                    nparams,
                                                                    dparams,
                                                                    oparams,
                                                                    qae_params,
                                                                    save_dir,
                                                                    load_params,
                                                                    rseed,
                                                                    plot_bloch,
                                                                    n_threads
                                                                ))
    n_workers = min(mp.cpu_count(), len(args_list))

    with mp.Pool(processes=n_workers) as pool:
        # starmap will call execute_job(*args_tuple) for each entry in job_args
        pool.starmap(execute_job, args_list, chunksize=1)
    

    # jobs = []
    # for dat_name in dat_names:
    #     for input_type in input_type_ls:
    #         dparams = DParams(dat_name, input_type, n_train, n_test)
    #         for n_layers in n_layers_ls:
    #             for scramb in scramb_ls:
    #                 for n_pj_qubits in n_pj_ls:
    #                     for type_evol in type_evol_ls:
    #                         if scramb == 'random' and type_evol == 'trotter':
    #                             print(f'Skip {scramb}-{type_evol}')
    #                         else:
    #                             for delta_t in delta_ls:
    #                                 nparams = NParams(gen_circuit_type = gen_circuit_type, n_qubits=n_qubits, n_ancilla=n_ancilla, n_layers=n_layers, n_diff_steps=n_diff_steps, \
    #                                                 scramb=scramb, type_evol=type_evol, n_pj_qubits=n_pj_qubits, delta_t=delta_t)
    #                                 for lr in lrs:
    #                                     for dist_type in dist_types:
    #                                         oparams = OParams(lr=lr, n_outer_epochs=n_outer_epochs, dist_type=dist_type)
    #                                         for rseed in rseeds:
    #                                             p = mp.Process(target=execute_job, args=(args.bin, nparams, dparams, oparams, save_dir, load_params, rseed, plot_bloch, n_threads))
    #                                             jobs.append(p)
        
    # # Start the process
    # for p in jobs:
    #     p.start()

    # # Ensure all processes have finished execution
    # for p in jobs:
    #     p.join()