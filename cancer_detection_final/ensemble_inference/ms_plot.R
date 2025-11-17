# Packages
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/pred_out_100125_union2/"

# Create a vector of folder names
folders <- paste0("MSI_traintf", format(seq(0.0, 0.9, by=0.1), nsmall = 1), "_Transfer_MIL_uni2")

# Initialize an empty list to store data frames
all_perf_data <- list()

# Loop through each folder and read the file
for (folder in folders) {
  file_path <- file.path(proj_dir, folder, "CV_performance", "TF0.0_perf_CV_AVG_SD.csv")
  if (file.exists(file_path)) {
    perf_df <- read.csv(file_path)
    #perf_df <- perf_df[which(perf_df$X == "OPX_TCGA_TEST_and_NEP_ALL"),]
    perf_df["Training_TF"] <- gsub("traintf", "", strsplit(folder, split = "_")[[1]][2])
    all_perf_data[[folder]] <- perf_df
  } else {
    warning(paste("File not found in", folder))
  }
}

# Combine all available data into a single data frame
combined_perf <- do.call(rbind, all_perf_data)
combined_perf <- combined_perf[, c("X","Training_TF", "AUC", "PR_AUC", "Recall", "Specificity","NPV","Precision","False_Positive_Rate")]
combined_perf <- combined_perf[-which(combined_perf$X == "OPX_TCGA_valid"),]
combined_perf <- combined_perf[-which(combined_perf$X == "OPX_TCGA_TEST"),]
#combined_perf <- combined_perf[-which(combined_perf$X == "NEP_ALL"),]
#combined_perf <- combined_perf[-which(combined_perf$X == "OPX_TCGA_TEST_and_NEP_ALL"),]


combined_perf <- combined_perf %>%
  mutate(
    X = case_when(
      X == "NEP_ALL" ~ "NEP (External Validation)",
      X == "OPX_TCGA_TEST_and_NEP_ALL" ~ "All",
      X == "OPX_test" ~ "UW (Hold-Out)",
      X == "TCGA_test" ~ "TCGA (Hold-Out)",
      X == "NEP_st_ALL" ~ "NEP (External Validation)",
      X == "OPX_union_test" ~ "UW (Hold-Out)",
      X == "TCGA_union_test" ~ "TCGA (Hold-Out)",
      #X == "OPX_TCGA_valid" ~ "UW + TCGA (Cross-Validation)",
      #X == "OPX_TCGA_TEST" ~ "All (Hold-Out)"
      TRUE ~ X  # keep other values unchanged
    )
  )

unique(combined_perf$X)

# View summary
summary(combined_perf)


#Plot


# --- 1. Prepare the data ---
library(dplyr)
library(stringr)
library(ggplot2)

# --- Build an AUC-only plotting frame, parsing mean ± sd ---
df_auc <- combined_perf %>%
  mutate(Training_TF = as.numeric(Training_TF),
         AUC = as.character(AUC)) %>%
  transmute(
    X, Training_TF,
    raw = AUC,
    raw_clean = str_replace_all(AUC, " ", "")
  ) %>%
  mutate(
    mean_str = str_split_fixed(raw_clean, "±|\\+/-", 2)[, 1],
    sd_str   = str_split_fixed(raw_clean, "±|\\+/-", 2)[, 2],
    mean     = suppressWarnings(as.numeric(mean_str)),
    sd       = suppressWarnings(as.numeric(sd_str)),
    # if no "±" present and it's already numeric, fill mean from raw
    mean = ifelse(is.na(mean), suppressWarnings(as.numeric(raw)), mean)
  )

# --- Order cohorts in legend and plot ---
df_auc$X <- factor(df_auc$X, levels = c(
  #"All (Hold-Out)",
  "All",
  "NEP (External Validation)",
  "UW (Hold-Out)",
  "TCGA (Hold-Out)"
  #"UW + TCGA (Cross-Validation)"
))

# --- Define a color-blind-friendly palette (Okabe–Ito) ---
custom_colors <- c(
  "All" = "#007B5E",                            # darker teal-green (from #009E73)
  "NEP (External Validation)" = "#C67C00",         # deeper golden orange (from #E69F00)
  "UW (Hold-Out)" = "#005A9C",                          # darker blue (from #0072B2)
  "TCGA (Hold-Out)" = "#A44500"                        # deeper red-orange (from #D55E00)
  # "UW + TCGA (Cross-Validation)" = "#666666"           # optional muted gray
)


# --- Define line styles for additional contrast ---
line_styles <- c(
  "All" = "solid",
  "NEP (External Validation)" = "longdash",
  "UW (Hold-Out)" = "dashed",
  "TCGA (Hold-Out)" = "dotdash"
  #"UW + TCGA (Cross-Validation)" = "twodash"
)

# --- Define point shapes for color + shape distinction ---
point_shapes <- c(
  "All" = 16,    # solid circle
  "NEP (External Validation)" = 17,  # solid triangle
  "UW (Hold-Out)" = 15,   # solid square
  "TCGA (Hold-Out)" = 18 # solid diamond
  #"UW + TCGA (Cross-Validation)" = 8  # star
)

# # --- Plot: one figure, one line per cohort, with error bars ---
# ggplot(df_auc, aes(
#   x = Training_TF,
#   y = mean,
#   color = X,
#   linetype = X,
#   shape = X
# )) +
#   geom_line(size = 1.3) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
#                 width = 0.015, alpha = 0.7, linewidth = 0.6, na.rm = TRUE) +
#   scale_color_manual(values = custom_colors) +
#   scale_linetype_manual(values = line_styles) +
#   scale_shape_manual(values = point_shapes) +
#   scale_x_continuous(
#     breaks = seq(0, 1.0, 0.1),
#     labels = function(x) sprintf("%.1f", x)
#   ) +
#   coord_cartesian(ylim = c(0.45, 1)) +
#   labs(
#     x = "Training Fraction (TF)",
#     y = "AUC (mean ± SD)",
#     color = "Cohort",
#     linetype = "Cohort",
#     shape = "Cohort",
#     title = "AUC vs Training Fraction by Cohort"
#   ) +
#   theme_classic(base_size = 15) +
#   theme(
#     legend.position = "right",
#     legend.title = element_text(size = 12, face = "bold"),
#     legend.text = element_text(size = 11),
#     plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
#     axis.text = element_text(size = 11, color = "black"),
#     axis.title = element_text(size = 13, face = "bold"),
#     panel.grid.major = element_line(color = "grey90", size = 0.3)
#   )




# prepare ymin/ymax, clipped to [0,1] and ordered for ribbons
dfp <- df_auc %>%
  arrange(X, Training_TF) %>%
  mutate(
    ymin = pmax(0, mean - sd),
    ymax = pmin(1, mean + sd)
  )

 p <- ggplot(dfp, aes(x = Training_TF, y = mean, color = X, linetype = X, shape = X)) +
  # subtle SD band (behind lines)
  geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = X),
              alpha = 0.10, colour = NA, show.legend = FALSE) +
  geom_line(size = 1.3) +
  geom_point(size = 3, stroke = 0.8) +
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  scale_linetype_manual(values = line_styles) +
  scale_shape_manual(values = point_shapes) +
  scale_x_continuous(breaks = seq(0, 1.0, 0.1),
                     labels = function(x) sprintf("%.1f", x)) +
  coord_cartesian(ylim = c(0.0, 1)) +
  labs(#x = "Cancer-pixel fraction threshold for weighted spatial sampling during training", 
       x = "Fraction of Cancer Pixels", #\n (weighted sampling during training)",
       y = "AUROC (mean ± SD)", color = "Cohort",
       linetype = "Cohort", shape = "Cohort",
       title = "") +
  theme_classic(base_size = 16)+
   theme(
     axis.title.x = element_text(size = 18, face = "bold", vjust = -0.2),  # bigger x-axis title
     axis.title.y = element_text(size = 18, face = "bold", vjust = 1.5),   # bigger y-axis title
     axis.text.x  = element_text(size = 18, color = "black"),              # larger x tick labels
     axis.text.y  = element_text(size = 18, color = "black"),              # larger y tick labels
     legend.position = "top",
     legend.direction = "horizontal",
     legend.box = "horizontal",
     legend.title = element_text(size = 18, face = "bold"),
     legend.text = element_text(size = 18),
     #plot.title = element_text(size = 20, face = "bold", hjust = 0.5),
     #plot.subtitle = element_text(size = 14, color = "gray30", hjust = 0.5),
     panel.grid.major = element_line(color = "grey90", size = 0.3)
   )+
   guides(color = guide_legend(nrow = 2),
          linetype = guide_legend(nrow = 2),
          shape = guide_legend(nrow = 2))
 
 print(p)
 ggsave(
   filename = "AUC_vs_TrainingFraction_by_Cohort_WithNEP.png",  # output filename
   plot = last_plot(),      # saves your most recent ggplot
   dpi = 600,               # 600 DPI for publication
   width = 8,               # width in inches
   height = 6,              # height in inches
   units = "in",            # specify inches for consistency
   bg = "white"             # white background (avoids transparency issues)
 )
