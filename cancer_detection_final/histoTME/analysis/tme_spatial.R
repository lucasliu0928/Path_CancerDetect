
library(ggplot2)
library(dplyr)
library(tidyr)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
histo_data_dir <- paste0(proj_dir, "0_HistoTME/TME_Spatial/TF0.9/uni2/")
cancer_data_dir <- paste0(proj_dir, "2_cancer_detection/OPX/IMSIZE250_OL0/")
sample_id <- "OPX_075"

#read
df <- read.csv(paste0(histo_data_dir, sample_id, "_5fold.csv"))
df <- df[,-1]
tile_info_df <- read.csv(paste0(cancer_data_dir, sample_id, "/ft_model/", sample_id, "_TILE_TUMOR_PERC.csv"))
tile_info_df <- tile_info_df %>%
  mutate(TILE_XY_INDEXES = gsub("[()]", "", TILE_XY_INDEXES)) %>%  # remove parentheses
  separate(TILE_XY_INDEXES, into = c("x", "y"), sep = ", ") %>%   # split by comma + space
  mutate(across(c(y, y), as.integer))                             # convert to integers

comb_df <- merge(df, tile_info_df, by = c('x','y'), all.y=TRUE)

# If duplicates exist, average them
df_avg <- comb_df %>%
  group_by(pred_map_location) %>%
  summarise(Checkpoint_inhibition = mean(Checkpoint_inhibition), .groups = "drop")


# Extract numbers
df_clean <- df_avg %>%
  mutate(pred_map_location = gsub("[()]", "", pred_map_location)) %>%  # remove parentheses
  separate(pred_map_location, into = c("x_start", "x_end", "y_start", "y_end"), sep = ", ") %>%
  mutate(across(c(x_start, x_end, y_start, y_end), as.numeric))


# # Plot heatmap
# ggplot(df_avg, aes(x = x, y = y, fill = Checkpoint_inhibition)) +
#   geom_tile() +
#   scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
#   theme_minimal() +
#   labs(title = "Heatmap of Checkpoint_inhibition",
#        x = "x",
#        y = "y",
#        fill = "Checkpoint_inhibition")


# Plot rectangles colored by Checkpoint_inhibition
ggplot(df_clean) +
  geom_rect(
    aes(xmin = y_start, xmax = y_end, ymin = x_start, ymax = x_end, fill = Checkpoint_inhibition),
    color = NA
  ) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    na.value = "gray"   # NA values shown as gray
  ) +
  coord_fixed() +
  scale_y_reverse() +  # optional, for image-like coords
  theme_minimal() +
  labs(
    title = "Heatmap of Checkpoint_inhibition",
    x = "X", y = "Y", fill = "Checkpoint_inhibition"
  )
