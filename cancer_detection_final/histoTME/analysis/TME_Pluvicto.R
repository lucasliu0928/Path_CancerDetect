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

#################### Dir ####################
cohort <- "Pluvicto_Pretreatment_bx"
label_var <- "Responder"
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/"
tme_dir   <- paste0(proj_dir, "intermediate_data/0_HistoTME/TME/TF0.0/")
label_dir <- paste0(proj_dir, "data/MutationCalls/",cohort, "/")

#################### Load ####################
tme_df   <- read.csv(paste0(tme_dir, cohort, "_predictions_uni2.csv"), check.names = FALSE)

if (cohort == "Pluvicto_Pretreatment_bx") {
  label_df <- read.xlsx(paste0(label_dir, "R_NR_labels_v1.xlsx"))
  colnames(label_df)[which(colnames(label_df) == "image")] <- "ID"
  colnames(label_df)[which(colnames(label_df) == "R_NR")] <- "Responder"
  
} else{
  label_df <- read.xlsx(paste0(label_dir, "TMA_sample_mapping.xlsx"))
  colnames(label_df)[which(colnames(label_df) == "Sample")] <- "ID"
  label_df <- label_df[which(label_df[,"Group"] == "pluvicto"),]
}


comb_df  <- merge(tme_df, label_df, by = "ID")

table(comb_df$Responder)

#################### Plot ####################
# Columns of interest
cols <- c("MHCI", "MHCII", "Coactivation_molecules", "Effector_cells",
          "T_cells", "T_cell_traffic", "NK_cells", "B_cells", "M1_signatures",
          "Th1_signature", "Antitumor_cytokines", "Checkpoint_inhibition",
          "Macrophage_DC_traffic", "T_reg_traffic", "Treg", "Th2_signature",
          "Macrophages", "Neutrophil_signature", "Granulocyte_traffic",
          "MDSC_traffic", "MDSC", "Protumor_cytokines", "Proliferation_rate",
          "EMT_signature", "Matrix", "Matrix_remodeling", "Endothelium",
          "CAF", "Angiogenesis")


# ---- First dataset (sh_df) ----
data_matrix1 <- as.matrix(comb_df[, cols])
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
                silent = FALSE)   # prevents plotting immediately
ph1





###########2nd way:
# Extract label vector and coerce to factor
labels <- as.factor(comb_df[, label_var])

# Build feature matrix (numeric only)
feature_df <- comb_df[, cols]
feature_df[] <- lapply(feature_df, function(x) as.numeric(as.character(x)))
feature_mat <- as.matrix(t(feature_df))  # rows = features, cols = samples

# ---- Map labels to readable names (adjust mapping if needed) ----
# Assumes "1" = Responder, "0" = Non-responder; tweak if your meaning differs
label_names <- ifelse(as.character(labels) %in% c("1", "Responder"),
                      "Responder", "Non-responder")
label_names <- factor(label_names, levels = c("Non-responder", "Responder"))

# ---- Order: Non-responder first, then Responder ----
ord <- order(label_names)
feature_mat  <- feature_mat[, ord, drop = FALSE]
labels_ord   <- label_names[ord]

# ---- Annotation for columns ----
annotation_col <- data.frame(Label = labels_ord)
rownames(annotation_col) <- colnames(feature_mat)

ann_colors <- list(Label = c("Responder" = "darkgreen",
                             "Non-responder" = "salmon"))

# ---- Add a vertical gap between groups & keep columns unclustered ----
gap_pos <- sum(labels_ord == "Non-responder")

p <- pheatmap(
  feature_mat,
  scale = "row",
  cluster_rows = TRUE,
  cluster_cols = TRUE,        # <-- keep the label-based separation
  treeheight_row = 0,
  treeheight_col = 0,
  annotation_col = annotation_col,
  annotation_colors = ann_colors,
  #gaps_col = gap_pos,          # vertical divider between groups
  show_colnames = FALSE,
  color = colorRampPalette(c("navy", "white", "firebrick3"))(50)
)

p




#3rd way
library(ComplexHeatmap)
library(circlize)

# feature_mat and label_names from above
col_split <- label_names  # factor with desired order

Heatmap(
  feature_mat,
  name = "z",
  color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
  cluster_rows = TRUE,
  cluster_columns = TRUE,       # clusters within each split
  column_split = col_split,     # <-- separates by label
  show_column_names = FALSE,
  top_annotation = HeatmapAnnotation(
    Label = col_split,
    col = list(Label = c("Responder" = "darkgreen",
                         "Non-responder" = "salmon"))
  ),
  column_title = NULL
)

############################################
#Coor
###########
df <- comb_df[,c("Responder",cols)]
colnames(df)[which(colnames(df) == "Responder")] <- "Label"

# Ensure Label is a factor
df$Label <- factor(df$Label, levels = c("Nonresponder", "Responder"))

# Convert Label to numeric (0 = Non-responder, 1 = Responder)
df$Label_numeric <- ifelse(df$Label == "Responder", 1, 0)

# Select only numeric columns (features)
feature_cols <- df %>%
  select(-Label) %>%
  select(where(is.numeric)) %>%
  select(-Label_numeric) %>%
  colnames()

# Compute correlations between each feature and the Label
cor_results <- sapply(feature_cols, function(x) {
  cor(df[[x]], df$Label_numeric, method = "spearman", use = "complete.obs")
})

# Convert to data frame
cor_df <- data.frame(
  Feature = names(cor_results),
  Spearman_correlation = as.numeric(cor_results)
) %>%
  arrange(desc(Spearman_correlation))

# Print sorted correlations
print(cor_df)

# Optional: visualize
library(ggplot2)
ggplot(cor_df, aes(x = reorder(Feature, Spearman_correlation),
                   y = Spearman_correlation,
                   fill = Spearman_correlation)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white") +
  theme_minimal() +
  labs(title = "Correlation between features and response (Spearman)",
       x = "Feature", y = "Correlation (Responder = 1)")

# Packages
# Load packages
# Load libraries
# Load libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)

# --- INPUT ---
# df must have:
#   df$Label: "Responder" / "Nonresponder"
#   numeric feature columns

df$Label <- factor(df$Label, levels = c("Nonresponder", "Responder"))

# Get numeric feature names
feature_cols <- df %>%
  select(-Label) %>%
  select(where(is.numeric)) %>%
  colnames()

# Define star labeling function
stars <- function(p) {
  case_when(
    is.na(p)     ~ "",
    p < 0.001    ~ "***",
    p < 0.01     ~ "**",
    p < 0.05     ~ "*",
    p < 0.1      ~ "·",
    TRUE         ~ ""
  )
}

# --- Loop through each feature and save plot ---
for (feat in feature_cols) {
  
  # Subset data for this feature
  d <- df %>% select(Label, all_of(feat)) %>%
    rename(Score = all_of(feat))
  
  # Compute t-test (raw p-value)
  r <- d %>% filter(Label == "Responder") %>% pull(Score)
  n <- d %>% filter(Label == "Nonresponder") %>% pull(Score)
  tt <- tryCatch(t.test(r, n, var.equal = FALSE), error = function(e) NULL)
  pval <- if (!is.null(tt)) unname(tt$p.value) else NA_real_
  star <- stars(pval)
  
  # For label placement
  y_max <- max(d$Score, na.rm = TRUE)
  y_label <- y_max + 0.05 * (abs(y_max) + 1e-9)
  
  # Create plot
  p <- ggplot(d, aes(x = Label, y = Score, fill = Label)) +
    geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.8) +
    geom_jitter(width = 0.15, alpha = 0.5, size = 1) +
    geom_text(aes(x = 1.5, y = y_label, label = star),
              inherit.aes = FALSE, size = 6, fontface = "bold") +
    scale_fill_manual(values = c("Nonresponder" = "#d62728", "Responder" = "#2ca02c")) +
    theme_minimal(base_size = 13) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "none",
      plot.title = element_text(face = "bold")
    ) +
    labs(
      title = paste0(feat, " (p = ", signif(pval, 3), ")"),
      subtitle = "Raw p-value (Welch t-test)",
      x = NULL,
      y = "Score"
    )
  
  # Save one PNG per feature
  fname <- paste0("boxplot_", feat, ".png")
  ggsave(filename = fname, plot = p, width = 5, height = 4, dpi = 300)
  
  message("Saved: ", fname)
}
