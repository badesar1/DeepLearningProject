# DeepLearningProject

This repository contains all code used to train models for each of the 3 objectives described in the project report. Note that training will not run out of the box, original training was performed on a SLURM cluster and some paths are hard coded. I also do not include the data used for training here, if you would like the data please feel free to reach out via email.

Objective 1)
training can be run with the command
`python3 train_fgatir_3dpatch.py --preprocess True --num_of_layers 20 --patchSize 70 --fname P3d70 --outf logs/FGtraining_P3d70_MSE_b64_l20_testAGnoise_nB_PS`

Objective 2)
`python3 train_fgatir_2dpatch_ricloss.py --preprocess True --num_of_layers 20 --patchSize 50 --fname P2d50 --outf logs/FGtraining_P2d50_MSE_b64_l20_testARnoise_Ricloss_nl75`

objective 3)
`python3 train_dwi.py --preprocess True --num_of_layers 20 --fname p50c5_3 --outf logs/DWItraining_P50dc5_MSE_b64_l20_nl50_6`
