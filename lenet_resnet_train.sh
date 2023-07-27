device=0
subj=1
roi_names=("prf-visualrois" "floc-bodies" "floc-faces" "floc-places" "floc-words" "streams" "unknown")

for roi in "${roi_names[@]}"
do 
    CUDA_VISIBLE_DEVICES=$device python training_lenet.py --subj=$subj --roi=$roi --lenet --log_comet --layer=1
done