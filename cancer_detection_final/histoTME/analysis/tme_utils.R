library(ggplot2)
library(tidyr)
library(pheatmap)
library(dplyr)
library(RColorBrewer)
library(ggplotify)  # to convert pheatmap -> ggplot
library(dendsort)
library(tidyverse)
library(ggpubr)
library(rstatix)


plot_feature_comparison <- function(tme_data, cohort, target_group, x_label){
  
  if (is.na(cohort) ==  F){
    #Cohort
    study_df <- tme_data[which(tme_data[,"COHORT"] == cohort),]
  }else{
    study_df <- tme_data
  }
  
  #rownames(study_df) <- study_df[,"SAMPLE_ID"]
  
  #study_df
  study_df <- study_df[,c(target_group, selected_cols)]
  study_df[,target_group] <- as.factor(study_df[,target_group])
  
  
  
  
  #Feature cols
  selected_cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
                     "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
                     "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
                     "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
                     "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
                     "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
                     "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
                     "CAF", "Angiogenesis")
  
  
  
  df_long <- study_df %>%
    pivot_longer(
      cols = all_of(selected_cols),
      names_to = "Feature",
      values_to = "Value"
    )
  
  
  if ("MSI" %in% target_group){
    df_long <- df_long %>%
      mutate(!!sym(target_group) := recode_factor(
        .data[[target_group]],
        "1" = "MSI +",
        "0" = "MSI -"
      ))
    comps <- list(c("MSI -", "MSI +"))
  }else{
    df_long <- df_long %>%
      mutate(!!sym(target_group) := recode_factor(
        .data[[target_group]],
        "1" = "High",
        "0" = "Low"
      ))
    comps <- list(c("High", "Low"))
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


#Get FP and FN
plot_heatmap_TME <- function(tme_data, cohort, out_dir){
  
  #Plot data
  if (is.na(cohort)){
    plot_df <- tme_data
  }else{
   plot_df <- tme_data[which(tme_data[,"COHORT"] == cohort),]
  }
  rownames(plot_df) <- plot_df[,"SAMPLE_ID"]
  
  #Get FP and FN
  n_FP <- length(which((plot_df$majority_class == 1) & (plot_df$MSI_POS == 0)))
  n_FN <- length(which((plot_df$majority_class == 0) & (plot_df$MSI_POS == 1)))
  
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
  feature_mat <- as.matrix(t(plot_df[ ,selected_cols]))
  
  
  # build top annotations (one column per bar)
  annotation_col <- data.frame(
    MSI_POS = factor(plot_df[, "MSI_POS"]),
    PRED_MSI_POS = factor(plot_df[, "majority_class"])
  )
  rownames(annotation_col) <- colnames(feature_mat)  
  
  # colors for each annotation (levels -> colors)
  ann_colors <- list(
    MSI_POS = c("0" = "#2ECC71", "1" = "#E67E22"),
    PRED_MSI_POS = c("0" = "#2ECC71", "1" = "#E67E22")
  )
  
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
    silent = TRUE,
  )
  print(p)
  save_heatmap(p, paste0(out_dir, cohort, "_histTME_heatmap.png"))
}



save_heatmap <- function(pheatmap_obj, filename, width = 10, height = 6, dpi = 300) {
  gg <- ggplotify::as.ggplot(pheatmap_obj$gtable)
  gg <- gg + theme_void(base_size = 12) + theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(2, 2, 2, 2)  # tight margins
  )
  ggsave(filename, gg, width = width, height = height, dpi = dpi, bg = "white")
}







plot_hffeature_comparison <- function(tme_data, cohort, target_group, x_label, selected_cols, topK =10 ,alpha = 0.05){
  
  # --- subset ---
  if (!is.na(cohort)) {
    study_df <- tme_data[which(tme_data[,"COHORT"] == cohort),]
  } else {
    study_df <- tme_data
  }
  
  study_df <- study_df[, c(target_group, selected_cols)]
  study_df[, target_group] <- as.factor(study_df[, target_group])
  
  # --- long format ---
  df_long <- study_df %>%
    pivot_longer(
      cols = all_of(selected_cols),
      names_to = "Feature",
      values_to = "Value"
    )
  
  # --- recode labels & set comparison ---
  if ("MSI" %in% target_group) {
    df_long <- df_long %>%
      mutate(!!rlang::sym(target_group) := dplyr::recode_factor(
        .data[[target_group]], "1" = "MSI +", "0" = "MSI -"
      ))
    comps <- list(c("MSI -", "MSI +"))
  } else {
    df_long <- df_long %>%
      mutate(!!rlang::sym(target_group) := dplyr::recode_factor(
        .data[[target_group]], "1" = "High", "0" = "Low"
      ))
    comps <- list(c("High", "Low"))
  }
  
  # --- compute per-feature Wilcoxon p-values & filter to significant features ---
  pvals <- df_long %>%
    group_by(Feature) %>%
    reframe(
      p = tryCatch(
        wilcox.test(Value ~ .data[[target_group]])$p.value,
        error = function(e) NA_real_
      )
    ) %>%
    filter(!is.na(p)) %>%
    slice_min(order_by = p, n = topK)
  
  sig_feats <- pvals %>%
    pull(Feature) %>%
    unique()
  
  df_sig <- df_long %>% filter(Feature %in% sig_feats)
  
  if (length(sig_feats) == 0) {
    stop(sprintf("No features are significant at FDR < %.02f.", alpha))
  }
  
  # --- plot only significant features ---
  p <- ggplot(df_sig, aes(x = !!rlang::sym(target_group), y = Value)) +
    geom_boxplot(width = 0.6, fill = "grey80", color = "black") +
    facet_wrap(~ Feature, scales = "free_y") +
    stat_compare_means(
      comparisons = comps,
      method = "wilcox.test",
      label = "p.signif",
      label.y.npc = 0.98,
      bracket.size = 0.6,
      tip.length = 0.01,
      hide.ns = TRUE   # hide non-significant labels (should be none after filtering)
    ) +
    labs(x = x_label, y = "Value") +
    theme_bw() +
    theme(
      axis.text = element_text(size = 14),
      axis.title = element_text(size = 14, face = "bold"),
      strip.text = element_text(size = 6, face = "bold"),
      legend.position = "none",
      strip.background = element_rect(fill = "grey90", color = NA)
    )
  
  return(p)
}
