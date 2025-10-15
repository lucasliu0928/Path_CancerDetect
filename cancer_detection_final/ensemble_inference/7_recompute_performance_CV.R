library(ggplot2)
library(dplyr)
library(tidyr)
library(dplyr)
library(yardstick)
library(tibble)
load_label <- function(cohort, cancer_thres){
  label_file <- file.path(label_dir,cohort, paste0("TFT",cancer_thres),"train_test_split.csv")
  label_df <- read.csv(label_file, stringsAsFactors = F)
  label_df['COHORT'] <- cohort
  colnames(label_df)[which(colnames(label_df) == "MSI_POS")] <- "MSI"
  
  return(label_df)
}

compute_logit_adjustment <- function(label_freq, tau){
  adjustments <- log(label_freq^(tau) + 1e-12)
  return(adjustments)
}



library(caret)
library(pROC)
library(PRROC)
library(tibble)

library(caret)
library(pROC)
library(PRROC)
library(tibble)

compute_classification_metrics <- function(
    df,
    truth_col = "True_y",
    prob_col  = "mean_adj_prob",
    pred_col  = "majority_class",
    positive  = "1"   # label treated as the positive class
) {
  # Pull columns
  actual    <- factor(df[[truth_col]], levels = c("0","1"))
  predicted <- factor(df[[pred_col]],  levels = c("0","1"))
  # If inputs are numeric 0/1, coerce to character -> factor
  if (!all(levels(actual) %in% c("0","1"))) {
    actual    <- factor(as.character(df[[truth_col]]), levels = c("0","1"))
    predicted <- factor(as.character(df[[pred_col]]),  levels = c("0","1"))
  }
  prob <- df[[prob_col]]
  
  # Confusion matrix
  cm <- confusionMatrix(predicted, actual, positive = positive)
  tbl <- cm$table
  # With confusionMatrix(data=predicted, reference=actual):
  # rows = Prediction, cols = Reference
  TP <- as.numeric(tbl["1","1"])
  TN <- as.numeric(tbl["0","0"])
  FP <- as.numeric(tbl["1","0"])
  FN <- as.numeric(tbl["0","1"])
  
  # AUCs
  roc_auc_value <- as.numeric(auc(roc(actual, prob)))  # ROC AUC
  
  pr_obj <- pr.curve(
    scores.class0 = prob[actual == positive],         # positives' scores
    scores.class1 = prob[actual != positive],         # negatives' scores
    curve = FALSE
  )
  pr_auc_value <- pr_obj$auc.integral                  # PR AUC
  
  # Conf-matrix metrics (positive class)
  recall      <- as.numeric(cm$byClass["Sensitivity"])
  specificity <- as.numeric(cm$byClass["Specificity"])
  precision   <- as.numeric(cm$byClass["Pos Pred Value"])  # PPV
  npv         <- as.numeric(cm$byClass["Neg Pred Value"])
  fpr         <- 1 - specificity
  acc         <- as.numeric(cm$overall["Accuracy"])
  
  # F-beta scores from precision/recall
  f_beta <- function(p, r, beta = 1) {
    if (is.na(p) || is.na(r) || (p + r) == 0) return(NA_real_)
    b2 <- beta^2
    (1 + b2) * p * r / (b2 * p + r)
  }
  F1 <- f_beta(precision, recall, beta = 1)
  F2 <- f_beta(precision, recall, beta = 2)
  F3 <- f_beta(precision, recall, beta = 3)
  
  # Assemble one-row data frame with metrics in columns
  out <- tibble::tibble(
    ROC_AUC            = roc_auc_value,
    PR_AUC             = pr_auc_value,
    Recall             = recall,
    Precision          = precision,
    NPV                = npv,
    Specificity        = specificity,
    False_Positive_Rate= fpr,
    ACC                = acc,
    F1                 = F1,
    F2                 = F2,
    F3                 = F3,
    TP = TP, TN = TN, FP = FP, FN = FN
  )
  
  return(out)
}

library(pROC)
library(dplyr)
library(tibble)

find_best_cutoff_youden <- function(df,
                                    truth_col = "True_y",
                                    prob_col  = "mean_adj_prob",
                                    positive_label = 1) {
  # --- Prepare inputs ---------------------------------------------------------
  actual_raw <- df[[truth_col]]
  prob       <- df[[prob_col]]
  
  # Coerce truth to a factor with levels "0","1" (positive = "1")
  # Works whether original is 0/1 numeric, character, or factor
  to01 <- function(x, pos = positive_label) {
    if (is.factor(x)) x <- as.character(x)
    if (is.character(x)) {
      as.integer(x == as.character(pos))
    } else {
      as.integer(x == pos)
    }
  }
  actual01 <- to01(actual_raw, pos = positive_label)
  actual_f <- factor(actual01, levels = c(0, 1), labels = c("0", "1"))
  
  # Guard: need >1 unique probability value for ROC to be defined
  if (length(unique(na.omit(prob))) < 2) {
    return(tibble(
      thresh = NA_real_, Sensitivity = NA_real_, Specificity = NA_real_,
      Youden_J = NA_real_, TP = NA_integer_, FP = NA_integer_,
      TN = NA_integer_, FN = NA_integer_, ROC_AUC = NA_real_
    ))
  }
  
  # --- ROC & AUC --------------------------------------------------------------
  roc_obj <- roc(response = actual_f,
                 predictor = prob,
                 levels   = c("0", "1"), # "1" is positive
                 direction = ">",        # higher prob => positive
                 quiet = TRUE)
  
  roc_auc_value <- as.numeric(auc(roc_obj))
  
  # --- Best cutoff by Youden's J ----------------------------------------------
  # coords returns a data.frame when transpose = FALSE
  best_df <- coords(
    roc_obj,
    x = "best",
    best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity"),
    transpose = FALSE
  ) %>% as.data.frame()
  
  # If multiple candidates tie, break ties by:
  # 1) max J, then 2) closest to sens == spec (balanced), then 3) smallest threshold
  best_row <- best_df %>%
    mutate(Youden_J = sensitivity + specificity - 1,
           tie_bal  = abs(sensitivity - specificity)) %>%
    arrange(desc(Youden_J), tie_bal, threshold) %>%
    slice(1)
  
  thresh <- as.numeric(best_row$threshold)
  sens   <- as.numeric(best_row$sensitivity)
  spec   <- as.numeric(best_row$specificity)
  J      <- as.numeric(best_row$Youden_J)
  
  # --- Confusion counts at this threshold -------------------------------------
  pred01 <- as.integer(prob >= thresh)
  TP <- sum(pred01 == 1 & actual01 == 1, na.rm = TRUE)
  FP <- sum(pred01 == 1 & actual01 == 0, na.rm = TRUE)
  TN <- sum(pred01 == 0 & actual01 == 0, na.rm = TRUE)
  FN <- sum(pred01 == 0 & actual01 == 1, na.rm = TRUE)
  
  tibble(
    thresh = thresh,
    Sensitivity = sens,
    Specificity = spec,
    Youden_J = J,
    TP = TP, FP = FP, TN = TN, FN = FN,
    ROC_AUC = roc_auc_value
  )
}



################################################################################
#User Input
################################################################################
outcome_folder <- "pred_out_100125_check_0.98PREV" #pred_out_100125, pred_out_100125_check, pred_out_100125_check_0.98PREV
mutation <- "MSI"
cancer_threshold <- "0.0"


################################################################################
#Dir
################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data"
label_dir <- file.path(proj_dir,"3C_labels_train_test")


################################################################################
# Load the label df to compute train / validation label freq
################################################################################
label_df_opx <- load_label("OPX",cancer_threshold)
label_df_tcga <- load_label("TCGA_PRAD",cancer_threshold)
label_df_nep <- load_label("Neptune",cancer_threshold)
label_df_opx_tcga <- do.call(rbind,list(label_df_opx,label_df_tcga))


################################################################################
#Compute CV performance avg += sd
################################################################################
folds <- c(0,1,2,3,4)
perf_list <- list()
for (i in 1:length(folds)){
  cur_fold <- folds[i]
  
  #Load prediction df
  pref_file <- file.path(proj_dir, outcome_folder, mutation, paste0("FOLD",cur_fold), paste0("perf_", cancer_threshold),"after_finetune_performance.csv")
  pref_df <- read.csv(pref_file, stringsAsFactors = F)
  pref_df["MODEL"] <- paste0("FOLD",cur_fold)
  perf_list[[i]] <- pref_df
}

all_perf_df <- do.call(rbind, perf_list)

#Get current cutoff for each fold
valid_perf_df <- all_perf_df[which(all_perf_df[,"X"] == "OPX_TCGA_valid"),]


#CV performance
cv_perf <- all_perf_df %>%
  group_by(X) %>%
  summarise(
    across(
      where(is.numeric) & !last_col(),
      list(
        mean = ~mean(.x, na.rm = TRUE),
        sd = ~sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    )
  ) %>%
  rowwise() %>%
  mutate(
    across(
      ends_with("_mean"),
      ~ {
        colname <- sub("_mean$", "", cur_column())
        mean_val <- .x
        sd_val <- get(paste0(colname, "_sd"))
        sprintf("%.2f ± %.2f", mean_val, sd_val)
      },
      .names = "{sub('_mean$', '', .col)}"
    )
  ) %>%
  select(X, !ends_with(c("_mean", "_sd"))) %>%
  ungroup()
write.csv(cv_perf,file.path(proj_dir, outcome_folder, mutation,"CV_performance","perf_CV_AVG_SD.csv"))


################################################################################
#Load prediction df
################################################################################
folds <- c(0,1,2,3,4)

pred_list <- list()
for (i in 1:length(folds)){
  cur_fold <- folds[i]
  
  #Load prediction df
  pred_file <- file.path(proj_dir, outcome_folder, mutation, paste0("FOLD",cur_fold), paste0("predictions_", cancer_threshold),"after_finetune_prediction.csv")
  pred_df <- read.csv(pred_file, stringsAsFactors = F)
  
  #select columns
  pred_df <- pred_df[,c("SAMPLE_ID","adj_prob_1","Pred_Class_adj","True_y")]
  pred_df["MODEL"] <- paste0("FOLD",cur_fold)
  pred_list[[i]] <- pred_df
  
}

all_pred_df <- do.call(rbind, pred_list)
all_pred_df<- all_pred_df[!duplicated(all_pred_df),]


################################################################################
#Ensemble model
################################################################################
ensemble_pred_df <- all_pred_df %>%
  group_by(SAMPLE_ID) %>%
  summarise(
    mean_adj_prob = mean(adj_prob_1, na.rm = TRUE),
    majority_class = as.integer(names(which.max(table(Pred_Class_adj)))),
    True_y = first(True_y) 
  )

ensemble_pred_df <- as.data.frame(ensemble_pred_df)
ensemble_pred_df[,"COHORT"] <- NA
ensemble_pred_df[which(grepl("OPX",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "OPX"
ensemble_pred_df[which(grepl("TCGA",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "TCGA"
ensemble_pred_df[which(grepl("NEP",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "NEP"
df_ensemble_opx <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "OPX"),]
df_ensemble_tcga <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "TCGA"),]
df_ensemble_nep <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "NEP"),]



res_opx <- compute_classification_metrics(df_ensemble_opx, truth_col = "True_y", prob_col = "mean_adj_prob",pred_col = "majority_class")
res_opx["CORHOT"] <- "OPX"
res_tcga <- compute_classification_metrics(df_ensemble_tcga, truth_col = "True_y", prob_col = "mean_adj_prob",pred_col = "majority_class")
res_tcga["CORHOT"] <- "TCGA"
res_nep <- compute_classification_metrics(df_ensemble_nep, truth_col = "True_y", prob_col = "mean_adj_prob",pred_col = "majority_class")
res_nep["CORHOT"] <- "NEPTUNE"


################################################################################
#If ensemble using majority class performance
#'@NOTE: ignore ROCAUC PrAUC, that is for mean_adj_prob, 
################################################################################
ensemble_majority_class_perf <- do.call(rbind, list(res_opx, res_tcga, res_nep))
ensemble_majority_class_perf <- ensemble_majority_class_perf[, !(names(ensemble_majority_class_perf) %in% c("ROC_AUC","PR_AUC"))] #aucs are computed using avg prob not majority

write.csv(ensemble_majority_class_perf,file.path(proj_dir, outcome_folder, mutation,"ensemble_performance","perf_ensemble.csv"))


# ################################################################################
# # Use avg predicted prob to find best threshold
# # redo logit adj for nep
# ################################################################################
# folds <- c(0,1,2,3,4)
# tau <- 0.1 # for logit adjustment, if tau = 0, means do not do adj
# 
# pred_list <- list()
# pref_list <- list()
# for (i in 1:length(folds)){
#   cur_fold <- folds[i]
#   
#   #Load prediction df
#   pred_file <- file.path(proj_dir, outcome_folder, mutation, paste0("FOLD",cur_fold), paste0("predictions_", cancer_threshold),"after_finetune_prediction.csv")
#   pred_df <- read.csv(pred_file, stringsAsFactors = F)
#   idxes<- which(grepl("NEP",pred_df[,"SAMPLE_ID"]))
#   pred_df <- pred_df[idxes,]
#   
#   #current threshold
#   cur_threshod <- valid_perf_df[which(valid_perf_df[,"MODEL"] == paste0("FOLD",cur_fold)),"best_thresh"]
#   
#   # If need to recompute adjustment
#   # for each fold, compute validation label feq
#   #idxes <- which(label_df_opx_tcga[,paste0("FOLD",cur_fold)] == "VALID")
#   #label_freq <- table(label_df_opx_tcga[idxes,mutation])/length(idxes)
#   if (tau > 0 ){
#       label_freq <- c(0.97,0.02)
#       #Compute logit adjustment
#       logit_adjustments <- compute_logit_adjustment(label_freq,tau)
#     
#       #Applied logit adjust to prediction output
#       pred_df[,"adj_logit_0"] <- pred_df[,"logit_0"] - logit_adjustments[1]
#       pred_df[,"adj_logit_1"] <- pred_df[,"logit_1"] - logit_adjustments[2]
#       
#       pred_df <- pred_df %>%
#         rowwise() %>%
#         mutate(
#           adj_prob_0 = exp(adj_logit_0) / (exp(adj_logit_0) + exp(adj_logit_1)),
#           adj_prob_1 = exp(adj_logit_1) / (exp(adj_logit_0) + exp(adj_logit_1))
#         ) %>%
#         ungroup()
#       pred_df<- as.data.frame(pred_df)
#       
#       #recode predicted class
#       pred_df[,"Pred_Class_adj"] <- 0
#       pred_df[which(pred_df[,"adj_prob_1"] >= cur_threshod),"Pred_Class_adj"] <- 1
#   }
#   
#   #select columns
#   pred_df <- pred_df[,c("SAMPLE_ID","adj_prob_1","Pred_Class_adj","True_y")]
#   pred_df["MODEL"] <- paste0("FOLD",cur_fold)
#   pred_list[[i]] <- pred_df
#   
#   
#   #fold perfomernace after adj
#   res_nep_cv <- compute_classification_metrics(pred_df, truth_col = "True_y", prob_col = "adj_prob_1",pred_col = "Pred_Class_adj")
#   res_nep_cv["MODEL"] <- paste0("FOLD",cur_fold)
#   pref_list[[i]] <- res_nep_cv
#   
# }
# 
# all_pred_df <- do.call(rbind, pred_list)
# all_pred_df_nep <- all_pred_df
# all_perf_df_nep <- do.call(rbind, pref_list)
# 
# 
# cv_perf_nep <- all_perf_df_nep %>%
#   summarise(
#     across(
#       where(is.numeric) & !last_col(),
#       list(
#         mean = ~mean(.x, na.rm = TRUE),
#         sd = ~sd(.x, na.rm = TRUE)
#       ),
#       .names = "{.col}_{.fn}"
#     )
#   ) %>%
#   rowwise() %>%
#   mutate(
#     across(
#       ends_with("_mean"),
#       ~ {
#         colname <- sub("_mean$", "", cur_column())
#         mean_val <- .x
#         sd_val <- get(paste0(colname, "_sd"))
#         sprintf("%.2f ± %.2f", mean_val, sd_val)
#       },
#       .names = "{sub('_mean$', '', .col)}"
#     )
#   ) %>%
#   ungroup()
# 
# 
# ################################################################################
# #TODO: REcompute CV for all cohort
# ################################################################################
# folds <- c(0,1,2,3,4)
# 
# pred_list <- list()
# for (i in 1:length(folds)){
#   cur_fold <- folds[i]
#   
#   #Load prediction df
#   pred_file <- file.path(proj_dir, outcome_folder, mutation, paste0("FOLD",cur_fold), paste0("predictions_", cancer_threshold),"after_finetune_prediction.csv")
#   pred_df <- read.csv(pred_file, stringsAsFactors = F)
#   
#   #select columns
#   pred_df <- pred_df[,c("SAMPLE_ID","adj_prob_1","Pred_Class_adj","True_y")]
#   pred_df["MODEL"] <- paste0("FOLD",cur_fold)
#   pred_list[[i]] <- pred_df
#   
# }
# 
# all_pred_df <- do.call(rbind, pred_list)
# 
# ################################################################################
# #Ensemble model
# ################################################################################
# ensemble_pred_df <- all_pred_df %>%
#   group_by(SAMPLE_ID) %>%
#   summarise(
#     mean_adj_prob = mean(adj_prob_1, na.rm = TRUE),
#     majority_class = as.integer(names(which.max(table(Pred_Class_adj)))),
#     True_y = first(True_y) 
#   )
# 
# ensemble_pred_df <- as.data.frame(ensemble_pred_df)
# res_nep <- compute_classification_metrics(ensemble_pred_df, truth_col = "True_y", prob_col = "mean_adj_prob",pred_col = "majority_class")
# res_nep["CORHOT"] <- "NEPTUNE"
# ensemble_majority_class_perf <- do.call(rbind, list(res_opx, res_tcga, res_nep))
# ensemble_majority_class_perf <- ensemble_majority_class_perf[, !(names(ensemble_majority_class_perf) %in% c("ROC_AUC","PR_AUC"))] #aucs are computed using avg prob not majority
# 
# write.csv(ensemble_majority_class_perf,file.path(proj_dir, outcome_folder, mutation,"ensemble_performance","perf_ensemble_updatedneptune_logitadj.csv"))
# 
# 
# ################################################################################################################################################################
# #For nep, split data into 5 folds, compute AOCAUC and PR_AUC, and find the best cutoff points
# ################################################################################################################################################################
# ensemble_pred_df <- ensemble_pred_df[,c("SAMPLE_ID","mean_adj_prob","True_y")]
# set.seed(123)
# 
# # create 5 folds (stratified by True_y)
# fold_list <- createFolds(ensemble_pred_df$True_y, k = 5, list = TRUE, returnTrain = FALSE)
# 
# # add 5 columns: fold1 ... fold5
# df_split <- ensemble_pred_df
# 
# for (i in 1:5) {
#   test_idx <- fold_list[[i]]
#   df_split[[paste0("FOLD", i-1)]] <- ifelse(seq_len(nrow(df_split)) %in% test_idx, "TEST", "TRAIN")
# }
# 
# 
# folds <- c(0,1,2,3,4)
# using_train <- FALSE
# best_cutoffs <- list()
# res_list <- list()
# for (i in 1:length(folds)){
#   cur_fold <- folds[i]
#   cur_train <- df_split[which(df_split[,paste0("FOLD", cur_fold)] == "TRAIN"),]
#   cur_test <- df_split[which(df_split[,paste0("FOLD", cur_fold)] == "TEST"),]
#   
# 
#   #threshold datasets
#   if (using_train == TRUE){
#     find_threshold_df <- cur_train
#     test_threshold_df <- cur_test
#   }else{
#     find_threshold_df <- cur_test
#     test_threshold_df <- cur_train
#   }
#   
#   #Use the small amount data to find best cutoff
#   best<- find_best_cutoff_youden(find_threshold_df, truth_col = "True_y", prob_col = "mean_adj_prob")
#   best$fold <- i
#   best_cutoffs[[i]] <- best
# 
#   #Prediction use the best threshold 
#   test_threshold_df[,"predicted_class"] <- 0
#   test_threshold_df[which(test_threshold_df[,"mean_adj_prob"] >= best$thresh),"predicted_class"] <- 1
#   
#   res <- compute_classification_metrics(test_threshold_df, truth_col = "True_y", prob_col = "mean_adj_prob",pred_col = "predicted_class")
#   res<- as.data.frame(res)
#   res$thresh <-  best$thresh
#   res$FOLD <- as.character(cur_fold)
#   res$n_pos <- length(which(test_threshold_df[,"True_y"] == 1))
#   res$n_neg <- length(which(test_threshold_df[,"True_y"] == 0))
#   
#   
#   res_list[[i]]<- res
#   print(res)
# }
# 
# final_resd <- do.call(rbind, res_list)
# 
# cv_perf_nep <- final_resd %>%
#   summarise(across(
#     where(is.numeric),
#     list(mean = ~mean(.x, na.rm = TRUE), sd = ~sd(.x, na.rm = TRUE)),
#     .names = "{.col}_{.fn}"
#   )) %>%
#   rowwise() %>%
#   mutate(
#     across(
#       ends_with("_mean"),
#       ~{
#         base <- sub("_mean$", "", cur_column())
#         mean_val <- .x
#         sd_val <- get(paste0(base, "_sd"))
#         sprintf("%.3f ± %.3f", mean_val, sd_val)
#       },
#       .names = "{sub('_mean$', '', .col)}"
#     )
#   ) %>%
#   select(!ends_with(c("_mean", "_sd"))) %>%
#   ungroup()
# 
# 
# 
