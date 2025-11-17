# Packages
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_100125_union2/"

# Create a vector of folder names
fe_methods <- c("virchow2","prov_gigapath","uni1","uni2","retccl")
folders <- paste0("MSI_traintf0.9_ABMIL_", fe_methods)

# Initialize an empty list to store data frames
all_perf_data <- list()

# Loop through each folder and read the file
for (folder in folders) {
  file_path <- file.path(proj_dir, folder, "CV_performance", "TF0.0_perf_CV_AVG_SD.csv")
  if (file.exists(file_path)) {
    perf_df <- read.csv(file_path)
    #perf_df <- perf_df[which(perf_df$X == "OPX_TCGA_TEST_and_NEP_ALL"),]
    perf_df["FE"] <- gsub("MSI_traintf0.9_ABMIL_", "", folder)
    
    
    all_perf_data[[folder]] <- perf_df
  } else {
    warning(paste("File not found in", folder))
  }
}

# Combine all available data into a single data frame
combined_perf <- do.call(rbind, all_perf_data)
combined_perf <- combined_perf[, c("X","FE","best_thresh", "AUC", "PR_AUC", "Recall", "Specificity","NPV","Precision","False_Positive_Rate")]

write.csv(combined_perf, file.path(proj_dir, "MSI_traintf0.9_ABMIL_AllFEmethods_perf.csv"))
