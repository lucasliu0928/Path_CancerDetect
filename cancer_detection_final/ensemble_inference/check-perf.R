library(ggplot2)
library(dplyr)
library(tidyr)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
data_dir <- paste0(proj_dir, "pred_out_100125/MSI/")

perf_list_before <- list()
perf_list_after <- list()
for (i in 1:5) {
  
  perf_dir <- paste0(data_dir, "FOLD", i-1, "/", "perf/")
  perf_df_before <- read.csv(paste0(perf_dir, "before_finetune_performance.csv"))
  perf_df_before['FOLD'] <- i -1
  perf_df_after <- read.csv(paste0(perf_dir, "after_finetune_performance.csv"))
  perf_df_after['FOLD'] <- i -1
  perf_list_before[[i]] <- perf_df_before
  perf_list_after[[i]] <- perf_df_after
}

perf_before <- do.call(rbind, perf_list_before)
perf_after <- do.call(rbind, perf_list_after)

