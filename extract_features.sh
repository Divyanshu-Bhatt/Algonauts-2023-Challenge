subj=(1 2 3 4 5 6 7 8)

for s in ${subj[@]}; do
    echo "Subject $s"
    CUDA_VISIBLE_DEVICES=1 python extract_features.py --subj=$s 
done