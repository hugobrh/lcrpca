cd /home/xxx/gprMax/
source /home/xxx/anaconda3/etc/profile.d/conda.sh
conda activate gprMax
python -m gprMax home/xxx/Documents/lcrpca_github/gprmax/dataset_creation.in -n 34 -gpu
python -m tools.outputfiles_merge home/xxx/Documents/lcrpca_github/gprmax/dataset_creation --remove-files
