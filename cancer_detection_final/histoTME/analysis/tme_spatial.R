
library(ggplot2)
library(dplyr)
library(tidyr)
library(rlang)


plot_tile_metric <- function(
    comb_df,
    sample_id,
    value_col = "T_cells",     # choose the column to visualize
    midpoint = NA,             # if NA, uses the median of the metric
    low = "blue", mid = "white", high = "red",  # colors
    width = 4, height = 4, dpi = 300,
    outdir = ""
) {

  #--- Average duplicates by pred_map_location for the chosen metric
  df_avg <- comb_df %>%
    group_by(pred_map_location) %>%
    summarise(.value = mean(.data[[value_col]], na.rm = TRUE), .groups = "drop")
  
  #--- Parse rectangle coordinates from "(x_start, x_end, y_start, y_end)"
  df_clean <- df_avg %>%
    mutate(pred_map_location = gsub("[()]", "", pred_map_location)) %>%
    separate(pred_map_location,
             into = c("x_start", "x_end", "y_start", "y_end"),
             sep = ", ", convert = TRUE)
  
  #--- Choose midpoint
  mp <- if (is.na(midpoint)) stats::median(df_clean$.value, na.rm = TRUE) else midpoint
  
  #--- Plot
  p <- ggplot(df_clean) +
    geom_rect(
      aes(xmin = y_start, xmax = y_end, ymin = x_start, ymax = x_end, fill = .value),
      color = NA
    ) +
    scale_fill_gradient2(low = low, mid = mid, high = high, midpoint = mp, na.value = "gray") +
    coord_fixed() +
    scale_y_reverse() +
    theme(
      panel.grid = element_blank(),       # remove grid
      axis.title = element_blank(),       # remove axis titles
      axis.text  = element_blank(),       # remove axis labels
      axis.ticks = element_blank(),       # remove axis ticks
      panel.background = element_rect(fill = "white", color = NA),
    ) +
    labs(fill = value_col)
  
  print(p)
  
  #--- Save
  outfile <- paste0(outdir, sample_id, "_", value_col, ".png")
  ggsave(outfile, plot = p, width = width, height = height, dpi = dpi, bg = "transparent")
  invisible(outfile)
}


############################################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
histo_data_dir <- paste0(proj_dir, "0_HistoTME/TME_Spatial/TF0.0/uni2/")
cohort_name <- "Neptune"
value_col <- "Checkpoint_inhibition" #Checkpoint_inhibition, T_cells
cancer_data_dir <- paste0(proj_dir, "2_cancer_detection/", cohort_name, "/IMSIZE250_OL0/")

ids <- c("OPX_129","OPX_167","OPX_207","OPX_198","OPX_216","OPX_263")
ids <- c("TCGA-HC-7747-01Z-00-DX1.1fd2554f-29b9-4a9e-9fe1-93a1bc29f319","TCGA-XK-AAIW-01Z-00-DX1.31B98849-4AA9-4AC1-A697-A0C4165976B5")
ids <- c("NEP-045PS1-1_HE_MH_03252024",
         "NEP-076PS1-1_HE_MH_03282024",
         "NEP-126PS1-1_HE_MH06032024",
         "NEP-159PS1-1_HE_MH_06102024",
         "NEP-212PS1-8_HE_MH_06282024",
         "NEP-212PS2_HE_MH_06282024",
         "NEP-280PS2-1_HE_MH_02072025")
ids <- c("NEP-346PS1-1_HE_MH_03202025",
         "NEP-377PS1-1_HE_MH_03212025")
for (sample_id in ids) {
#--- Read data
df <- read.csv(paste0(histo_data_dir, sample_id, "_5fold.csv"))
if (ncol(df) > 0) df <- df[, -1, drop = FALSE]  # drop first column if it's an index
# basic guard rails
if (!all(c("x", "y") %in% names(df))) {
  stop("Input CSV must contain columns named 'x' and 'y'. Found: ",
       paste(names(df), collapse = ", "))
}
if (!value_col %in% names(df)) {
  stop("Column '", value_col, "' not found. Available numeric columns: ",
       paste(names(df)[vapply(df, is.numeric, logical(1))], collapse = ", "))
}

# tile info
if (cohort_name == "TCGA_PRAD") {
  fold_name_df <- read.csv("/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/3C_labels_train_test/TCGA_PRAD/TFT0.0/train_test_split.csv")
  
  f_name <- fold_name_df$FOLDER_ID[which(fold_name_df$SAMPLE_ID == sample_id)]
}else{
  f_name <- sample_id
}

tile_info_df <- read.csv(
  paste0(cancer_data_dir, f_name, "/ft_model/", sample_id, "_TILE_TUMOR_PERC.csv")
) %>%
  mutate(TILE_XY_INDEXES = gsub("[()]", "", TILE_XY_INDEXES)) %>%
  separate(TILE_XY_INDEXES, into = c("x", "y"), sep = ", ") %>%
  mutate(across(c(x, y), as.integer))

#--- Merge on tile indices
comb_df <- merge(df, tile_info_df, by = c("x", "y"), all.y = TRUE)

#plot
plot_tile_metric(comb_df = comb_df, sample_id = sample_id, 
                 value_col = value_col,
                 outdir = histo_data_dir)
}



