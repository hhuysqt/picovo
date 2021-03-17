BEGIN	{
  read_time = 0
  process_time = 0
  nr_features = 0
  residuals = 0
  nr_iterations = 0
  nr_retrack_iterations = 0
  picoedge_time = 0
  feature_calc_time = 0
  post_process_time = 0
  nr_kf = 0
  printf "read\tPEdge\t3D feat\ttrack\tpost\ttotal\tnr feat\titers\tKF pct\n"
}
{
  FS = ","
  read_time += $1
  process_time += $2
  nr_features += $3
  residuals += $4
  nr_iterations += $5
  nr_retrack_iterations += $6
  picoedge_time += $7
  feature_calc_time += $8
  post_process_time += $9
  if ($6 != 0) {
    nr_kf += 1
  }
}
END	{
  read_time /= NR
  picoedge_time /= NR
  feature_calc_time /= NR
  post_process_time /= NR
  process_time /= NR
  nr_features /= NR
  nr_iterations /= NR
  tracking_time = process_time - picoedge_time - feature_calc_time - post_process_time
  printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.5f\n", read_time, 
    picoedge_time, feature_calc_time, tracking_time, post_process_time, 
    process_time, nr_features, nr_iterations, nr_kf/NR
}
