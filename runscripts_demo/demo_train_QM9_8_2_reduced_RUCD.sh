# To enable each single run on a single thread
export OMP_NUM_THREADS=1 #important for CPU & numpy
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
#export NUM_INTER_THREADS=1
#export NUM_INTRA_THREADS=1

#export NPROC=1 #to avoid multiple processes using the same GPU/CPU
# FOR JAX
# export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"


RUN=pyrun_train_QDM_mol_uncon.py
BIN=../source/main_gen_mol_uncon.py
SCRIPT_NAME=$(basename $0 .sh)
#FOLDER_NAME=$SCRIPT_NAME
FOLDER_NAME=demo_train_qm9_mol_8_2_reduced_RUCD

THREADS=1

SEEDS='0,1,2,3,4,5,6,7,8,9'

DATNAME='qm9'
ANSATZ='rxycz'
#ANSATZ='SU2-full'

NATOMS=8
NRINGS=2
NQUBITS=7
NANCILA=2 #for backward
N_PJ=2 #for forward

NTRAIN=2000
NTEST=4236
BATCH_SIZE=100

GEN_LAYERS='10'
EPOCHS=1001

BLOCH=0

DIST_TYPE='wass'
LR='0.001' # Learning rate
MAG='1.0'  # Magnitude of initial parameters
VENDI_LAMBDA='0.0' # Vendi loss lambda

SCRAMB='random'
EVOL='full'

USE_QAE=1
QAE_LATENT_DIM=4
QAE_LAYERS=20
QAE_EPOCHS=2000

#for delta in 0.0001 0.001 0.01
for delta in 10.0
do
for STEPS in 20
do
for INPUT in product
do
SAVE=../results_demo/$FOLDER_NAME/$DATNAME\_qubits_$NQUBITS\_dat_$NTEST\_npj_$N_PJ
# Using taskset to limit the CPU cores used by the script
taskset -c 0-9 python $RUN --bin $BIN --use_qae $USE_QAE --qae_latent $QAE_LATENT_DIM --qae_layers $QAE_LAYERS --qae_epochs $QAE_EPOCHS --batch_size $BATCH_SIZE --vendi_lambda $VENDI_LAMBDA --gen_circuit_type $ANSATZ --n_atoms $NATOMS --n_rings $NRINGS  --threads $THREADS --rseed $SEEDS --n_outer_epochs $EPOCHS --dat_name $DATNAME --bloch $BLOCH --n_diff_steps $STEPS --n_qubits $NQUBITS --n_ancilla $NANCILA --lr $LR --mag $MAG --save_dir $SAVE --input_type $INPUT --n_train $NTRAIN --n_test $NTEST --n_layers $GEN_LAYERS --dist_type $DIST_TYPE --scramb $SCRAMB --type_evol $EVOL --delta_t $delta --n_pj_qubits $N_PJ
done
done
done