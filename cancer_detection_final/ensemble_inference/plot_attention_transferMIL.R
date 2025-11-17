library(ggplot2)
library(dplyr)
library(tidyr)
library(scico)
library(scales)
library(arrow)

min_max_norm <- function(indata, col_name){
  min_v <- min(indata[,col_name], na.rm = TRUE)
  max_v <- max(indata[,col_name], na.rm = TRUE)
  indata[,col_name] <-  (indata[,col_name]  - min_v)/(max_v - min_v)
  
  return(indata)
}

########################################################################################################################
#User input
########################################################################################################################
infer_tf <- 0.0
plot_tf <- 0.0
train_folder <- "MSI_train_restrict_to_0.9sampling"
outcome_folder <- "pred_out_100125_test"



########################################################################################################################
#DIR
########################################################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
data_dir <- file.path(proj_dir, outcome_folder,train_folder, "ensemble_prediction","attention/")

########################################################################################################################
#Load attention
########################################################################################################################
att_df <- as.data.frame(read_parquet(paste0(data_dir, "TF", format(infer_tf, nsmall = 1), "_ensemble_attention.parquet")))

########################################################################################################################
#Plot
########################################################################################################################
# cond1 <- att_df[,"majority_class"] == att_df[,"True_y"]
# cond2 <- att_df[,"True_y"] == 1
# true_pos_ids <- unique(att_df[which(cond1 & cond2),"SAMPLE_ID"])

cond1 <- att_df[,"majority_class"] == att_df[,"True_y"]
cond2 <- att_df[,"True_y"] == 0
true_neg_ids <- unique(att_df[which(cond1 & cond2),"SAMPLE_ID"])
true_neg_ids <- true_neg_ids[144:160]
out_folder_name <- "True_negatives_"

for (sample_id in true_neg_ids){
    df <-  att_df[which(att_df[,"SAMPLE_ID"] == sample_id),]
    df <-  df[which(df[,"tumor_fraction"] >= plot_tf),]
    
    pred_prob <- unique(df[,"mean_adj_prob_votedclass"])

    # Get min/max for normalization
    df <- min_max_norm(df,"mean_att")
    
    p <- ggplot(df) +
      geom_rect(aes(xmin = y1, xmax = y2, ymin = x1, ymax = x2, fill = mean_att),
                color = NA) +
      scale_y_reverse() +
      coord_equal() +
      scale_fill_scico(
                       palette = "vik",
                       name = "Attention\nScore",
                       direction = 1)+
      # scale_fill_gradientn(
      #   colours = c("#2C7BB6","white", "#B2182B"),
      #   values  = c(0, 0.5, 1),
      #   name    = "Attention\nScore"
      # ) +
      theme_void() +
      #labs(title = paste0("Predicted Prob: ", format(round(pred_prob,2),nsmall = 2)), fill = "mean_att") +
      theme(
        legend.position = "none",
        legend.title = element_text(size = 6, face = "bold"),
        legend.text = element_text(size = 5),
        legend.key.height = unit(0.25, "cm"),
        legend.key.width  = unit(0.3, "cm"),
        plot.title = element_text(size = 8, hjust = 0.5),
        plot.margin = margin(2, 2, 2, 2),
        plot.background = element_rect(fill = "grey95", color = NA),
        panel.background = element_rect(fill = "grey95", color = NA)
      )
    print(p)
    ggsave(paste0(data_dir,out_folder_name, sample_id, ".png"), plot = p, width = 2.3, height = 3, dpi = 200)

}
