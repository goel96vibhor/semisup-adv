#!/bin/sh
#SBATCH --job-name=test
#SBATCH --output=slurm_out/job_output-test_%j.txt
#SBATCH --mail-user=rohit.sharma@euler.wacc.wisc.edu
#SBATCH --time=0-12:30:00
#SBATCH --gres=gpu:1
#SBATCH -p ppc
#SBATCH -p lianglab
#SBATCH --qos=lianglab_owner
module load python/3.7
#module load anaconda/3/2019.07
module load cuda/9.0 groupmods/cudnn/9.2
#module load groupmods/lianglab/cuda/9.2-dnn
STARTTIME=$(date +%s)
date

#python3 robust_self_training.py --model='resnet-20' --unsup_fraction=0.1
#python train_cifar10_vs_ti.py --base_model_path=rst_augmented/unsup_fraction_test/fraction_0.5/resnet-20/checkpoint-epoch50.pt --also-use-base-model=1 --output_dir=cifar10-vs-ti/ --use-old-detector=1 --dataset='tinyimages'

#python train_cifar10_vs_ti.py --detector_model_path rst_augmented/unsup_filtering_test/two_detector_filtering/resnet-20/checkpoint-epoch50.pt --detector-model resnet-20 --base_model_path=rst_augmented/unsup_fraction_test/fraction_0.5/resnet-20/checkpoint-epoch50.pt --also-use-base-model=1 --use-old-detector=1 --dataset='cifar10' --num_images=250000 --even_odd=1 --load_ti_head_tail=1 --store_to_dataframe=1 --output_dir=rst_augmented/unsup_filtering_test/two_detector_filtering/resnet-20/ --n_classes=10

#python train_cifar10_vs_ti.py --detector_model_path rst_augmented/unsup_filtering_test/two_detector_filtering_2/resnet-20/checkpoint-epoch50.pt --detector-model resnet-20 --base_model_path=rst_augmented/unsup_fraction_test/fraction_0.5/resnet-20/checkpoint-epoch50.pt --also-use-base-model=1 --use-old-detector=1 --dataset='cifar10' --num_images=250000 --even_odd=1 --load_ti_head_tail=1 --store_to_dataframe=1 --output_dir=rst_augmented/unsup_filtering_test/two_detector_filtering_2/resnet-20/ --n_classes=10

#python filtered_training.py --model='resnet-20' --test_name='unsup_filtering_test/two_detector_filtering_2' --filter_unsup_data=1 --unsup_std_deviations=1 --unsup_fraction=0.5 --use_two_detector_filtering=1

python filtered_training.py --model='resnet-20' --test_name='example_sampling/d1_d2_diff_l1_weighing_0.1' --filter_unsup_data=1 --unsup_std_deviations=1 --unsup_fraction=0.5 --use_two_detector_filtering=1 --use_example_sampling=1 --example_weight_alpha=0.1 --use_distrib_selection=0 --random_split_version=2


ENDTIME=$(date +%s)
date
echo "Time taken: $(($ENDTIME - $STARTTIME)) seconds"
