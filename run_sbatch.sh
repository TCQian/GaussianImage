dataset=("HoneyBee" "Beauty" "Jockey")
testing_gaussians_1frames=(25000 35000)
for i in "${!dataset[@]}"; do
    for j in "${!testing_gaussians_1frames[@]}"; do
        sbatch run_gaussianimage.sh \
            --data_name "${dataset[$i]}" \
            --num_points "${testing_gaussians_1frames[$j]}" \
            --start_frame 0 \
            --num_frames 50
    done
done