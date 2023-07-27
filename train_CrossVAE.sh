
subj=1
device=0
roi_names=("prf-visualrois" "floc-bodies" "floc-faces" "floc-places" "floc-words" "streams" "unknown")

for s in "${subj[@]}"
do
    for roi_name in "${roi_names[@]}"
        do
            CUDA_VISIBLE_DEVICES=$device python train_kfold_mp.py  --log_in_comet --tag=CrossVAE_layer=3_clip_embeddings_${roi_name}_small_valset \
                                                            --subj=$s --num_workers=16 \
                                                            --device=cuda --epochs=1000 --batch_size=64 \
                                                            --lr=2e-5 --weight_decay=0.001 \
                                                            --layer=3 --patience=50 \
                                                            --annealing_iteration=3000 --annealing_how=cyclic \
                                                            --roi_data --name_roi=$roi_name
        done    
done