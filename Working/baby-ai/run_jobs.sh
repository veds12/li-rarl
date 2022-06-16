declare -a All_Round=(42 43 44 45 46)  

for seed in "${All_Round[@]}"
do
    sbatch submit_babyai.sh ${seed}
done