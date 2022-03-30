# for weight in -0.5 -0.1 0 0.1 0.5
for weight in -0.5 0 0.5
do
  for clfstr in transformer_correlation_input
  do
    for regioni in {2..2}
    do
      for dataprepstrs in climate coords climate+coords normalize+climate normalize+coords normalize+climate+coords
      do
        for wd in 0 1e-4
        do
          sbatch --requeue grid_job.sub --clf-strs $clfstr --data-prep-strs $dataprepstrs --weight $weight --generalization-region usa_g${regioni} --weight-decay $wd
        done
       done
     done
  done
done


for weight in 0
do
  for clfstr in transformer
  do
    for regioni in {2..2}
    do
      for dataprepstrs in normalize none
      do
        for wd in 0 1e-4
        do
          sbatch --requeue grid_job.sub --clf-strs $clfstr --data-prep-strs $dataprepstrs --weight $weight --generalization-region usa_g${regioni} --weight-decay $wd
        done
       done
     done
  done
done
