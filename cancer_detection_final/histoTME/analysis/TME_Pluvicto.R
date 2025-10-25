library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(ggpubr)
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) BiocManager::install("ComplexHeatmap")

library(tidyverse)
library(ComplexHeatmap)
library(circlize)
library(ggplot2)
library(tidyr)
library(pheatmap)
library(RColorBrewer)
library(ggplotify)  # to convert pheatmap -> ggplot
library(dendsort)
library(rstatix)

save_heatmap <- function(pheatmap_obj, filename, width = 10, height = 6, dpi = 300) {
  gg <- ggplotify::as.ggplot(pheatmap_obj$gtable)
  gg <- gg + theme_void(base_size = 12) + theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(2, 2, 2, 2)  # tight margins
  )
  ggsave(filename, gg, width = width, height = height, dpi = dpi, bg = "white")
}
plot_heatmap_TME <- function(tme_data, cohort, group_by_flag, out_dir){
  
  #Plot data
  rownames(tme_data) <- tme_data[,"ID"]
  
  #Feature cols
  selected_cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
                     "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
                     "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
                     "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
                     "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
                     "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
                     "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
                     "CAF", "Angiogenesis")
  
  # Build feature matrix (numeric only)
  feature_mat <- as.matrix(t(tme_data[ ,selected_cols]))
  
  
  # build top annotations (one column per bar)
  
  if ("death" %in% colnames(tme_data)){
      annotation_col <- data.frame(
        Responder = factor(tme_data[, "Responder"]),
        death = factor(tme_data[, "death"])
      )
      rownames(annotation_col) <- colnames(feature_mat)  
      
      # colors for each annotation (levels -> colors)
      ann_colors <- list(
        Responder = c(
          "Responder" = "#81C784",      # softer green
          "Nonresponder" = "#D55E00",   # warm amber/gold
          "NE_final" = "#FFB300"        # rich red
        ),
        death = c(
          "0" = "#CC79A7",              # gentle green for alive
          "1" = "#999999"               # deep orange-red for dead
        )
      )
  }else{
    annotation_col <- data.frame(
      Responder = factor(tme_data[, "Responder"])
    )
    rownames(annotation_col) <- colnames(feature_mat)  
    
    # colors for each annotation (levels -> colors)
    if ("NE_final" %in% unique(tme_data[, "Responder"])){
        ann_colors <- list(
          Responder = c(
            "Responder" = "#81C784",      # softer green
            "Nonresponder" = "#D55E00",   # warm amber/gold
            "NE_final" = "#FFB300"        # rich red
          ))
    }else{
      ann_colors <- list(
        Responder = c(
          "Responder" = "#81C784",      # softer green
          "Nonresponder" = "#D55E00"   # warm amber/gold
        )
        )
     }
    
  }
  
  
  if (group_by_flag == "None"){
    p <- pheatmap(
      feature_mat,
      annotation_col = annotation_col,          # <-- stacked bars on top
      annotation_colors = ann_colors,
      color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
      clustering_distance_rows = "euclidean",
      clustering_distance_cols = "euclidean",
      cluster_rows = FALSE,
      cluster_cols = TRUE, 
      clustering_method = "ward.D2",
      fontsize_row = 10,
      fontsize_col = 10,
      border_color = NA,
      show_colnames = FALSE,
      main = cohort,
    )
  }else if (group_by_flag == "Responder"){
    
    
    # group factor in desired display order
    group_var <- factor(tme_data$Responder,
                        levels = c("Responder", "Nonresponder", "NE_final"))
    
    # 2) Score each column to define the gradient direction (blue→red = low→high)
    #    Use column mean; replace with your preferred score if needed.
    col_score <- colMeans(feature_mat, na.rm = TRUE)
    
    # 3) For each group, order columns by ascending score (low→high)
    col_idx_by_group <- split(seq_len(ncol(feature_mat)), group_var)
    orders <- lapply(col_idx_by_group, function(idx) {
      if (length(idx) <= 1) return(idx)
      idx[order(col_score[idx], na.last = NA)]
    })
    
    # 4) Concatenate in group order and compute gaps
    ordered_cols <- unlist(orders, use.names = FALSE)
    
    gap_positions <- cumsum(vapply(col_idx_by_group, length, integer(1)))
    gap_positions <- gap_positions[gap_positions < ncol(feature_mat)]
    
    # 5) Reorder data + annotations
    feature_mat_ord    <- feature_mat[, ordered_cols, drop = FALSE]
    annotation_col_ord <- annotation_col[ordered_cols, , drop = FALSE]
    
    # 6) Plot: no within-group clustering, consistent gradient, visible group splits
    p <- pheatmap(
      feature_mat_ord,
      annotation_col    = annotation_col_ord,
      annotation_colors = ann_colors,
      color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
      cluster_cols = FALSE,            # <- no within-group clustering
      cluster_rows = TRUE,
      clustering_distance_rows = "euclidean",
      clustering_method = "ward.D2",
      gaps_col = gap_positions,
      show_colnames = FALSE,
      fontsize_row = 10,
      fontsize_col = 10,
      border_color = NA,
      main = paste(cohort, "- \n grouped by Responder")
    )
  }
  print(p)
  
  save_heatmap(p, paste0(out_dir, cohort, "_" ,group_by_flag, "_histTME_heatmap.png"))
}
plot_feature_comparison <- function(tme_data, cohort, target_group, x_label){
  
  
  
  rownames(tme_data) <- tme_data[,"ID"]
  
  selected_cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
                     "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
                     "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
                     "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
                     "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
                     "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
                     "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
                     "CAF", "Angiogenesis")
  
  study_df <- tme_data[,c(target_group,selected_cols)]
  
  #Exclude NA respond-er
  indx_to_remove <- which(study_df[,target_group] == "NE_final")
  if (length(indx_to_remove) > 0){
    study_df <- study_df[-indx_to_remove,]
  }
  
  # Ensure Label is a factor
  study_df[,target_group] <- factor(study_df[,target_group], levels = c("Nonresponder", "Responder"))
  
  
  #Long formart
  df_long <- study_df %>%
    pivot_longer(
      cols = all_of(selected_cols),
      names_to = "Feature",
      values_to = "Value"
    )
  
  
  if ("Responder" %in% target_group){
    comps <- list(c("Nonresponder", "Responder"))
  }
  
  
  
  # Plot
  p <- ggplot(df_long, aes(x = !!sym(target_group), y = !!sym("Value")))  +
    geom_boxplot(width = 0.6, fill = "grey80", color = "black") +  # keep outliers
    facet_wrap(~ Feature, scales = "free_y") +
    stat_compare_means(
      comparisons = comps,
      method = "wilcox.test",
      label = "p.signif",         # show exact p-value; use "p.signif" for stars
      label.y.npc = 0.98,         # put label near the top of each facet
      bracket.size = 0.6,
      tip.length = 0.01,
      hide.ns = FALSE
    ) +
    labs(x = x_label, y = "Value") +
    theme_bw() +
    theme(
      axis.text = element_text(size = 14),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 14, face = "bold"),
      legend.position = "none",
      strip.background = element_rect(fill = "grey90", color = NA)
    )
  
  return(p)
}
group_by_patient <- function(indata){
  
  #Feature cols
  selected_cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
                     "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
                     "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
                     "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
                     "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
                     "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
                     "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
                     "CAF", "Angiogenesis")
  other_cols <- setdiff(colnames(indata),c(selected_cols,"Patient"))
  
  df_avg <- indata %>%
    group_by(Patient) %>%
    summarise(
      across(all_of(selected_cols), ~ mean(.x, na.rm = TRUE)),
      across(all_of(other_cols), ~ first(.x)),
      .groups = "drop"
    )
  return(df_avg)
}
#################### Dir ####################
cohort <- "Pluvicto_TMA_Cores" #Pluvicto_Pretreatment_bx
label_var <- "Responder"
model <- "ensemble"
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/"
tme_dir   <- paste0(proj_dir, "intermediate_data/0_HistoTME/TME/TF0.0/")
label_dir <- paste0(proj_dir, "data/MutationCalls/",cohort, "/")
out_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/0_HistoTME/pluvicto_analysis/"

#################### Load ####################
#1.Load TME
tme_df   <- read.csv(paste0(tme_dir, cohort, "_", model,".csv"), check.names = FALSE)

#2.Load label
if (cohort == "Pluvicto_Pretreatment_bx") {
  label_df <- read.xlsx(paste0(label_dir, "R_NR_labels_v1.xlsx"))
  colnames(label_df)[which(colnames(label_df) == "image")] <- "ID"
  colnames(label_df)[which(colnames(label_df) == "R_NR")] <- "Responder"
  
} else{
  label_df <- read.xlsx(paste0(label_dir, "TMA_sample_mapping.xlsx"))
  colnames(label_df)[which(colnames(label_df) == "Sample")] <- "ID"
  label_df <- label_df[which(label_df[,"Group"] == "pluvicto"),]
  label_df <- label_df[-which(label_df$TMA %in%  c('TMA113A', 'TMA113B')),]
}

#3. Combine
comb_df  <- merge(tme_df, label_df, by = "ID")

#average per patient
if (cohort == "Pluvicto_TMA_Cores"){
  comb_df <- as.data.frame(group_by_patient(comb_df))
  comb_df <- comb_df %>%
    mutate(Responder = recode(Responder,
                                "Non-responder" = "Nonresponder",
                                "Responder" = "Responder"))
  
}

table(comb_df$Responder)

#4. Plot heatmap
plot_heatmap_TME(comb_df, cohort, "None", out_dir)
plot_heatmap_TME(comb_df, cohort, "Responder", out_dir)


#5. Comparison between groups
p <- plot_feature_comparison(comb_df, cohort, "Responder","")
ggsave(paste0(out_dir,"/", cohort, "_responder_TME_comparison.png"), p, width = 18, height = 20, dpi = 300)

