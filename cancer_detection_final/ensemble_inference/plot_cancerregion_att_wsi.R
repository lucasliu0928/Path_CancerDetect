library(magick)
library(ggplot2)
library(cowplot)
library(grid)

proj_dir <- "~/Desktop/figs/"

# --- Helper: trim & resize all images to same height ---
prep_img <- function(path, target_height = 1600) {
  image_read(path) |>
    image_trim() |>                          # remove white margins
    image_scale(paste0("x", target_height))  # match height
}

# --- Load and preprocess images ---
img_HE         <- prep_img(file.path(proj_dir, "OPX_167_low-res.png"))
img_cancer     <- prep_img(file.path(proj_dir, "OPX_167_cancer_prob.jpeg"))
img_attention  <- prep_img(file.path(proj_dir, "OPX_167.png"))

# --- Convert to rasterGrob ---
img_HE_raster_1        <- grid::rasterGrob(img_HE, interpolate = TRUE)
img_cancer_raster_1    <- grid::rasterGrob(img_cancer, interpolate = TRUE)
img_attention_raster_1 <- grid::rasterGrob(img_attention, interpolate = TRUE)

# --- Create panels (same as before) ---
base_theme <- theme_void() + theme(plot.margin = margin(0, 0, 0, 0))

p1 <- ggplot() +
  annotation_custom(img_HE_raster_1, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
  ggtitle("H&E WSI") +
  base_theme +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))

p2 <- ggplot() +
  annotation_custom(img_cancer_raster_1, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
  ggtitle("Cancer Region") +
  base_theme +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))

p3 <- ggplot() +
  annotation_custom(img_attention_raster_1, xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf) +
  ggtitle("MSI-H Attention Map") +
  base_theme +
  theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5))

# --- Combine with smaller gap ---
combined_plot <- plot_grid(
  p1, p2, p3,
  nrow = 1,
  align = "h",
  rel_widths = c(1, 1, 1)
)

# Save final aligned PNG
ggsave(
  filename = "Representative_MSIH_Case_equalHeight.png",
  plot = combined_plot,
  width = 10, height = 5, units = "in",
  dpi = 600, bg = "white"
)



library(cowplot)
library(ggplot2)

# Load your two PNGs
fig_A <- ggdraw() + draw_image("/Volumes/Lucas/mh_proj/mutation_pred/code_s/cancer_detection_final/ensemble_inference/AUC_vs_TrainingFraction_by_Cohort_WithNEP.png", scale = 1)
fig_B <- ggdraw() + draw_image("/Volumes/Lucas/mh_proj/mutation_pred/code_s/cancer_detection_final/ensemble_inference/Representative_MSIH_Case_equalHeight.png", scale = 1)



# Add labels
fig_A_labeled <- ggdraw(fig_A) +
  draw_label("(A) Performance vs. cancer fraction threshold", x = 0.02, y = 0.98, hjust = 0, vjust = 1,
             fontface = "bold", size = 12)

fig_B_labeled <- ggdraw(fig_B) +
  draw_label("(B) Example of true postive MSI-H case", x = 0.02, y = 0.98, hjust = 0, vjust = 1,
             fontface = "bold", size = 12)

# Combine with relative width control
final_combined <- plot_grid(
  fig_A_labeled, fig_B_labeled,
  nrow = 1,
  rel_widths = c(1.1, 1.2),  # make right figure wider
  align = "v"
)

library(cowplot)
library(grid)

# Add border to the combined figure
final_with_border <- final_combined +
  theme(
    plot.background = element_rect(color = "black", fill = "white", linewidth = 1.2),
    plot.margin = margin(3, 3, 3, 3)  # prevent clipping of the stroke
  )
# Save final output
ggsave(
  filename = "Final_Figure_AB_withNEP.png",
  plot = final_combined,
  width = 10, height = 4, units = "in",
  dpi = 300, bg = "white"
)
