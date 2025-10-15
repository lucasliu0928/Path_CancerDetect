library(ggplot2)
library(dplyr)
library(tidyr)

proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data/"
tf <- 0.9
data_dir <- paste0(proj_dir, "pred_out_100125/MSI/FOLD0/predictions/attention/")
data_dir <- paste0(proj_dir, "pred_out_100125/MSI/FOLD0/predictions_", tf, "/attention/")


ids1 <- c("OPX_129","OPX_167","OPX_198","OPX_207","OPX_216","OPX_263")
ids2 <- c("TCGA-HC-7747-01Z-00-DX1.1fd2554f-29b9-4a9e-9fe1-93a1bc29f319",
         "TCGA-XK-AAIW-01Z-00-DX1.31B98849-4AA9-4AC1-A697-A0C4165976B5")
ids3 <- c("NEP-045PS1-1_HE_MH_03252024",
         "NEP-076PS1-1_HE_MH_03282024",
         "NEP-126PS1-1_HE_MH06032024",
         "NEP-159PS1-1_HE_MH_06102024", #TF0.9
         "NEP-212PS1-8_HE_MH_06282024",
         "NEP-212PS2_HE_MH_06282024",
         "NEP-280PS2-1_HE_MH_02072025",
         "NEP-346PS1-1_HE_MH_03202025",
         "NEP-377PS1-1_HE_MH_03212025")


ids <- c(ids1, ids2, ids3)

for (sample_id in ids){
    df <- read.csv(paste0(data_dir, sample_id, "_att.csv"))
    
    # ---- Parse the stored_map_location ----
    # Remove parentheses and split into numeric coordinates
    # df has: stored_map_location, att
    coords <- do.call(rbind, strsplit(gsub("[()]", "", df$pred_map_location), ",\\s*"))
    coords <- apply(coords, 2, as.numeric)
    colnames(coords) <- c("x1","x2","y1","y2")
    df <- cbind(df, coords)
    
    # Get min/max for normalization
    # Get symmetric limits (so 0 is centered)
    library(scico)
    library(scales)
    
    #lim <- max(abs(df$att), na.rm = TRUE)
    min_v <- min(df$att, na.rm = TRUE)
    max_v <- max(df$att, na.rm = TRUE)
    
    df$att <-  (df$att - min_v)/(max_v - min_v)
    
    p <- ggplot(df) +
      geom_rect(aes(xmin = y1, xmax = y2, ymin = x1, ymax = x2, fill = att),
                color = NA) +
      scale_y_reverse() +
      coord_equal() +
      scico::scale_fill_scico(
        palette = "vik",   # perceptually uniform blue ↔ red
        #limits = c(-lim, lim)
      ) +
      theme_void() +
      labs(title = "", fill = "att") +
      #theme(legend.position = "none")
      theme(
        plot.background = element_rect(fill = "grey95", color = NA),
        panel.background = element_rect(fill = "grey95", color = NA)
      )
    print(p)
    ggsave(paste0(data_dir, sample_id, ".png"), plot = p, width = 4, height = 4, dpi = 300)

}