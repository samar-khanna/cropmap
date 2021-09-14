# for region in washington
for region in washington california us_south corn_belt new_york
do
    # for barycenter_method in EM_barycenter SG_barycenter soft_barycenter mean
    for barycenter_method in mean
    do
        # for samples in 10000
        for samples in 99999999999999
        do
            echo "$region $barycenter_method $samples"
            sbatch cpu.sub $region $barycenter_method $samples
        done
    done
done
