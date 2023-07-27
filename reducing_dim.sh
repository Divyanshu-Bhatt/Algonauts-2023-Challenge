roi=("prf-visualrois" "floc-bodies" "floc-faces" "floc-places" "floc-words" "streams" "unknown")
device=1
subj=(1 2 3 4 5 6 7 8)

for s in ${subj[@]}
do
    for r in ${roi[@]}
        do
            echo "Subject: $s ROI: $r"
            CUDA_VISIBLE_DEVICES=0 python reducing_dimension_vae.py --subj=$s --fmri="lh" --roi=$r &
            CUDA_VISIBLE_DEVICES=1 python reducing_dimension_vae.py --subj=$s --fmri="rh" --roi=$r
    done
done