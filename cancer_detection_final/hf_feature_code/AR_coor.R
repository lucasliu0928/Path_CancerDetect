library(openxlsx)
library(stringr)
library(dplyr)
library(ggplot2)
library(ggpubr)   # for stat_compare_means()



proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/"
label_dir <- paste0(proj_dir, "data/MutationCalls/TAN_TMA_Cores/")
feature_dir <- paste0(proj_dir, "data/TAN_WSI/UWTAN_HCfeatures/")

##########################################################################################
#Load Labels
##########################################################################################
#mutation_df1 <- read.xlsx(paste0(label_dir, "HaffnerAlleleStatusRerunComparisonsTruncal_revMCH.xlsx"), sheet = "HaffnerAlleleStatusRerun")
mutation_df_slide <- read.xlsx(paste0(label_dir, "HaffnerAlleleStatusRerunComparisonsTruncal_revMCH.xlsx"), sheet = "Sheet1")

#Core label
#map_df <- read.xlsx(paste0(label_dir, "TAN97_core_mappings2.xlsx")) 
#comb_df <- merge(map_df, mutation_df2, by.x = 'ptid', by.y = 'Sample')
##########################################################################################

##########################################################################################
slide_files <- list.files(feature_dir, full.names = T) #255
list_feat <- list()
for (i in 1:length(slide_files)){
  cur_f <- slide_files[i]
  df <- read.csv2(cur_f, sep = ",")
  list_feat[[i]] <- df
}
tile_hf_features <- do.call(rbind, list_feat)
tile_hf_features <- tile_hf_features %>% mutate(Sample = sub("[A-Za-z].*$", "", sub("(_.*)$", "", basename(dirname(tile)))))
tile_hf_features <- tile_hf_features[-which(tile_hf_features$Sample == '.'),]
length(unique(tile_hf_features$Sample)) #100

comb_df <- merge(tile_hf_features, mutation_df_slide, by = "Sample") #46

##########################################################################################
#Plot
#Nuclear Area = nuclear size
#Mean nuclear perimeter
#Triangular area
#Nuclear eccentricity
df$Size_Area_mean
df$Size_Perimeter_mean
df$density_distance_for_neighbors_2_mean_stdev
df$Shape_Eccentricity_mean
##########################################################################################
select_var <- "density_distance_for_neighbors_2_mean_stdev"
grp_var <- "AR_group"

plot_df <- comb_df[,c("Sample",select_var,"AR.coded","AR")]
plot_df[,select_var] <- as.numeric(plot_df[,select_var])
agg_df <- plot_df %>%
  group_by(Sample, AR.coded, AR) %>%
  summarise(
    "{select_var}" := mean(.data[[select_var]], na.rm = TRUE),
    .groups = "drop"
  )
agg_df <- as.data.frame(agg_df)
agg_df[,"AR_group"]   <- factor(agg_df[,"AR.coded"], levels = c(0,1), labels = c("none","mutation"))

# colors (match the figure)
cols <- c("mutation" = "#b2182b",   # red
          "none" = "#2166ac")   # blue
# place the p-value a bit above the tallest point
y_top <- max(agg_df[,select_var], na.rm = TRUE) * 1.05
y_positions <- c(y_top + 0.5, y_top + 2.5, y_top + 5.5)

# 1) compute test results
library(rstatix)
fmt_p <- function(p) ifelse(p < 1e-4,
                            formatC(p, format = "e", digits = 2),
                            formatC(p, format = "f", digits = 3))

stat_test <- agg_df %>%
  wilcox_test(reformulate("AR_group", response = select_var)) %>%
  add_significance() %>%
  add_xy_position(x = "AR_group") %>%
  mutate(
    p.format = fmt_p(p),
    label = paste0("p = ", p.format, p.signif)
  )

p <- ggplot(agg_df, aes(x = AR_group, y = .data[[select_var]],
                        color = AR_group, fill = AR_group)) +
  geom_boxplot(width = 0.55, alpha = 0.35, outlier.shape = NA) +
  geom_jitter(width = 0.12, size = 2, alpha = 0.85) +
  scale_color_manual(values = cols, name = "AR_group") +
  scale_fill_manual(values = cols,  name = "AR_group") +
  labs(x = "AR_group", y = select_var) +
  theme_classic(base_size = 12) +
  theme(legend.position = "right") +
  stat_pvalue_manual(
    stat_test,
    label = "label",
    xmin = "group1", xmax = "group2",
    y.position = "y.position",
    tip.length = 0.01
  )

p
ggsave(paste0(select_var, "_" ,grp_var, ".png"), plot = p, width = 6, height = 5, dpi = 300)

