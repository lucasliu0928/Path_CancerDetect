library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(ggpubr)


proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
tme_dir   <- paste0(proj_dir, "0_HistoTME/TME/TF0.0/")


sh_df <- read.csv(paste0(tme_dir,"precog_results_sh.csv"))
lucas_df <- read.csv(paste0(tme_dir,"PrECOG_predictions_uni2.csv"))
common_ids <- intersect(lucas_df$ID, sh_df$sample)
lucas_df <- lucas_df[which(lucas_df$ID %in% common_ids),]
rownames(lucas_df) <- NULL

lucas_df <- lucas_df[order(lucas_df$ID),]
rownames(lucas_df) <- lucas_df$ID
sh_df <- sh_df[match(lucas_df$ID, sh_df$sample), ]
rownames(sh_df) <- sh_df$sample

# Columns of interest
cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
          "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
          "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
          "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
          "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
          "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
          "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
          "CAF", "Angiogenesis")

# Initialize results
results <- data.frame(
  Column      = character(),
  Correlation = numeric(),
  P_value     = numeric(),
  stringsAsFactors = FALSE
)

# Loop through columns and compute correlation + p-value
for (col in cols) {
  test <- cor.test(lucas_df[[col]], sh_df[[col]], method = "pearson")
  results <- rbind(results, 
                   data.frame(Column = col,
                              Correlation = test$estimate,
                              P_value     = test$p.value))
}



# Plot correlations with p-values annotated
ggplot(results, aes(x = reorder(Column, Correlation), y = Correlation)) +
  geom_col(fill = "steelblue") +
  # geom_text(aes(label = sprintf("p=%.3f", P_value)), 
  #           vjust = -0.5, size = 3) +
  coord_flip() +
  theme_minimal() +
  labs(x = "TME",
       y = "Pearson Correlation")



library(pheatmap)
library(pheatmap)

# ---- First dataset (sh_df) ----
data_matrix1 <- as.matrix(sh_df[, cols])
t_matrix1 <- t(scale(data_matrix1))

# Run clustering once (don't draw yet)
ph1 <- pheatmap(t_matrix1,
                color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
                cluster_rows = TRUE,
                cluster_cols = TRUE,
                treeheight_row = 0,
                treeheight_col = 0,
                show_rownames = TRUE,
                show_colnames = TRUE,
                fontsize_row = 8,
                fontsize_col = 8,
                border_color = NA,
                silent = TRUE)   # prevents plotting immediately

# Extract order of rows/columns
row_order <- ph1$tree_row$order
col_order <- ph1$tree_col$order

# Plot first heatmap with consistent order
pheatmap(t_matrix1[row_order, col_order],
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
         main = "SH",
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize_row = 8,
         fontsize_col = 8,
         border_color = NA)


# ---- Second dataset (lucas_df) ----
data_matrix2 <- as.matrix(lucas_df[, cols])
t_matrix2 <- t(scale(data_matrix2))

# Plot second heatmap using the same order
pheatmap(t_matrix2[row_order, col_order],
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
         main = "LL",
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize_row = 8,
         fontsize_col = 8,
         border_color = NA)



# Combine the two datasets (assuming same samples & features)
# If they have different samples, keep them separate
data_matrix_lucas <- as.matrix(lucas_df[, cols])
data_matrix_sh    <- as.matrix(sh_df[, cols])

# Example: difference between the two datasets per sample
diff_matrix <- data_matrix_lucas - data_matrix_sh

# Compute sample–sample Euclidean distance on differences
sample_dist <- dist(diff_matrix, method = "euclidean")

# Convert to a matrix
sample_diff_matrix <- as.matrix(sample_dist)

# Plot heatmap
pheatmap(sample_diff_matrix,
         color = colorRampPalette(c("navy", "white", "firebrick3"))(50),
         main = "Sample Difference Heatmap",
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         treeheight_row = 10,
         treeheight_col = 10,
         show_rownames = TRUE,
         show_colnames = TRUE,
         fontsize_row = 8,
         fontsize_col = 8,
         border_color = NA)
