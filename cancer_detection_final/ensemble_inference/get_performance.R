library(ggplot2)
library(dplyr)
library(tidyr)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
data_dir <- paste0(proj_dir, "pred_out_100125/MSI/FOLD0/perf_")


tf_list <- seq(from = 0.0, to = 0.9, by = 0.1)

perf_list <- list()
for (i in 1:length(tf_list)){
  tf<- tf_list[i]
  
  pref_df <- read.csv(paste0(data_dir, format(round(tf, 1), nsmall = 1), "/after_finetune_performance_bootstraping_alltiles.csv"))
  pref_df['tf'] <- tf
  perf_list[[i]] <- pref_df
}

final_perf_df <- do.call(rbind,perf_list)
write.csv(final_perf_df, paste0(proj_dir, "pred_out_100125/MSI/FOLD0/perf_all_TFthresholds.csv"))
