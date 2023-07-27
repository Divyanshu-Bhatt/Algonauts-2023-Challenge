mkdir trained_models
cd trained_models
subj=(1 2 3 4 5 6 7 8)

for s in "${subj[@]}"
do
    mkdir subj_0$s
done
