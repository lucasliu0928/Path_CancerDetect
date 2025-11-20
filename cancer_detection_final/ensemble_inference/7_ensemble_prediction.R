library(ggplot2)
library(dplyr)
library(tidyr)
library(dplyr)
library(yardstick)
library(tibble)
library(caret)
library(pROC)
library(PRROC)
library(tibble)
library(dplyr)
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
  
  balanced_accuracy <- (recall + specificity) / 2
  
  # Assemble one-row data frame with metrics in columns
  out <- tibble::tibble(
    ROC_AUC            = roc_auc_value,
    PR_AUC             = pr_auc_value,
    Recall             = recall,
    Precision          = precision,
    NPV                = npv,
    Specificity        = specificity,
    False_Positive_Rate= fpr,
    balanced_accuracy  =  balanced_accuracy,
    ACC                = acc,
    F1                 = F1,
    F2                 = F2,
    F3                 = F3,
    TP = TP, TN = TN, FP = FP, FN = FN
  )
  
  return(out)
}



################################################################################
#User Input
################################################################################
outcome_folder <- "pred_out_100125_union2_check" #"pred_out_100125_union2" 
mutation <- "MSI"
model_folder<- "MSI_traintf0.9_Transfer_MIL_uni2/"
cancer_threshold_infer <- "0.0"


################################################################################
#Dir
################################################################################
proj_dir <- "/Volumes/Lucas/mh_proj/mutation_pred/intermediate_data"



################################################################################
#Load prediction df
################################################################################
folds <- c(0,1,2,3,4)

pred_list <- list()
for (i in 1:length(folds)){
  cur_fold <- folds[i]
  
  #Load prediction df
  pred_file <- file.path(proj_dir, outcome_folder,  model_folder, paste0("FOLD",cur_fold), paste0("predictions_", cancer_threshold_infer),"after_finetune_prediction.csv")
  pred_df <- read.csv(pred_file, stringsAsFactors = F)
  
  #select columns
  pred_df <- pred_df[,c("SAMPLE_ID","adj_prob_1","Pred_Class_adj","True_y")]
  pred_df["MODEL"] <- paste0("FOLD",cur_fold)
  pred_list[[i]] <- pred_df
  


}

all_pred_df <- do.call(rbind, pred_list)
all_pred_df<- all_pred_df[!duplicated(all_pred_df),]




################################################################################
#Ensemble Prediction
################################################################################
ensemble_pred_df <- all_pred_df %>%
  group_by(SAMPLE_ID) %>%
  summarise(
    mean_adj_prob = mean(adj_prob_1, na.rm = TRUE),
    majority_class = as.integer(names(which.max(table(Pred_Class_adj)))),
    mean_adj_prob_votedclass = mean(adj_prob_1[Pred_Class_adj == majority_class]),   # average only on voted class
    folds_voted = paste(MODEL[Pred_Class_adj == majority_class], collapse = ","),  # list folds voting that way
    True_y = first(True_y) 
  )


ensemble_pred_df <- as.data.frame(ensemble_pred_df)
ensemble_pred_df[,"COHORT"] <- NA
ensemble_pred_df[which(grepl("OPX",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "OPX"
ensemble_pred_df[which(grepl("TCGA",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "TCGA"
ensemble_pred_df[which(grepl("NEP",ensemble_pred_df[,"SAMPLE_ID"])),"COHORT"] <- "NEP"

new_dir_path <- file.path(proj_dir, outcome_folder,  model_folder,"ensemble_prediction")
if (!dir.exists(new_dir_path)) {
  # If it doesn't exist, create it
  dir.create(new_dir_path)
  message(paste("Directory '", new_dir_path, "' created successfully.", sep = ""))
} else {
  message(paste("Directory '", new_dir_path, "' already exists.", sep = ""))
}
write.csv(ensemble_pred_df, file.path(proj_dir, outcome_folder, model_folder,"ensemble_prediction",paste0("TF",cancer_threshold_infer, "_pred_ensemble.csv")))

#Ensemble prediction for each cohort
df_ensemble_opx   <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "OPX"),]
df_ensemble_tcga  <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "TCGA"),]
df_ensemble_nep   <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] == "NEP"),]
df_ensemble_opx_tcga   <- ensemble_pred_df[which(ensemble_pred_df[,"COHORT"] %in% c("OPX","TCGA")),]

################################################################################
#Compute Performance 
################################################################################
res_opx_tcga_nep <- compute_classification_metrics(ensemble_pred_df, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
res_opx_tcga_nep["CORHOT"] <- "OPX_TCGA_NEPTUNE"

res_opx_tcga <- compute_classification_metrics(df_ensemble_opx_tcga, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
res_opx_tcga["CORHOT"] <- "OPX_TCGA"

res_opx <- compute_classification_metrics(df_ensemble_opx, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
res_opx["CORHOT"] <- "OPX"
res_tcga <- compute_classification_metrics(df_ensemble_tcga, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
res_tcga["CORHOT"] <- "TCGA"
res_nep <- compute_classification_metrics(df_ensemble_nep, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
res_nep["CORHOT"] <- "NEPTUNE"



################################################################################
#If ensemble using majority class performance
#'@NOTE: ignore ROCAUC PrAUC, that is for mean_adj_prob_votedclass, 
################################################################################
ensemble_majority_class_perf <- do.call(rbind, list(res_opx, res_tcga, res_nep, res_opx_tcga, res_opx_tcga_nep))
ensemble_majority_class_perf <- ensemble_majority_class_perf[, !(names(ensemble_majority_class_perf) %in% c("ROC_AUC","PR_AUC"))] #aucs are computed using avg prob not majority


new_dir_path <- file.path(proj_dir, outcome_folder,  model_folder,"ensemble_performance")
if (!dir.exists(new_dir_path)) {
  # If it doesn't exist, create it
  dir.create(new_dir_path)
  message(paste("Directory '", new_dir_path, "' created successfully.", sep = ""))
} else {
  message(paste("Directory '", new_dir_path, "' already exists.", sep = ""))
}

write.csv(ensemble_majority_class_perf,file.path(proj_dir, outcome_folder, model_folder,"ensemble_performance",paste0("TF",cancer_threshold_infer, "_perf_ensemble.csv")))

    
    



##Combine all TF ensemble performance
cv_folder <-  file.path(proj_dir, outcome_folder, model_folder,"ensemble_performance")
csv_files <-  list.files(cv_folder, pattern = "\\.csv$")

cv_perf_list <- list()
for (i in 1:length(csv_files)){
  cur_df <- read.csv(file.path(cv_folder, csv_files[i]))
  cur_df <- cur_df[, -which(names(cur_df) == "X")]
  cur_df['TF'] <- gsub("_perf_ensemble.csv", "", csv_files[i])
  cv_perf_list[[i]] <- cur_df
}
combined_data <- do.call(rbind, cv_perf_list)
write.csv(combined_data, paste0(cv_folder, "/ALL_TF_ensemble_performance.csv"))


#For each threshold compute performance
thresholds <- seq(0,1,0.01)
res_list <- list()
for (i in 1 :length(thresholds)){
  th <- thresholds[i]
  pred_prob  <- ensemble_pred_df['mean_adj_prob_votedclass']
  ensemble_pred_df['new_class'] <- 0
  ensemble_pred_df[which(ensemble_pred_df['mean_adj_prob_votedclass'] >= th), 'new_class'] <- 1
  res <- compute_classification_metrics(ensemble_pred_df, 
                                                     truth_col = "True_y", 
                                                     prob_col = "mean_adj_prob_votedclass",
                                                     pred_col = "new_class")
  res['threshold'] <- th
  res_list[[i]] <- res
  
}
res_data <- do.call(rbind, res_list)
write.csv(res_data,file.path(proj_dir, outcome_folder, model_folder,"ensemble_performance",paste0("Thresholds_perf_ensemble.csv")))

################################################################################
#DCA
################################################################################

dca_continuous <- function(df,
                           prob_col = "mean_adj_prob_votedclass",
                           y_col    = "True_y",
                           thresholds = seq(0.01, 0.99, by = 0.01)) {
  p  <- df[[prob_col]]
  y  <- df[[y_col]]
  n  <- length(y)
  prev <- mean(y == 1)
  
  out <- data.frame(threshold = thresholds,
                    net_benefit_model = NA_real_,
                    net_benefit_all   = NA_real_,
                    net_benefit_none  = 0)
  
  for (i in seq_along(thresholds)) {
    t <- thresholds[i]
    
    treat_pos <- p >= t
    TP <- sum(treat_pos & y == 1) / n
    FP <- sum(treat_pos & y == 0) / n
    
    # Net benefit for the model
    out$net_benefit_model[i] <- TP - FP * (t / (1 - t))
    
    # Treat-all strategy
    TP_all <- prev
    FP_all <- 1 - prev
    out$net_benefit_all[i] <- TP_all - FP_all * (t / (1 - t))
  }
  
  out
}

# Run DCA and plot
dca_res <- dca_continuous(ensemble_pred_df)

plot(dca_res$threshold, dca_res$net_benefit_model, type = "l",
     xlab = "Threshold probability",
     ylab = "Net benefit")
lines(dca_res$threshold, dca_res$net_benefit_all, lty = 2)
abline(h = 0, lty = 3)   # treat-none
legend("topright",
       legend = c("Model", "Treat all", "Treat none"),
       lty = c(1, 2, 3), bty = "n")


library(rmda)


dca_model <- decision_curve(
  True_y ~ mean_adj_prob_votedclass,
  data = ensemble_pred_df,
  family = binomial(link = "logit"),
  thresholds = seq(0.01, 0.99, by = 0.01),   # can be coarser if you like
  confidence.intervals = TRUE,
  study.design = "cohort"
)


plot_decision_curve(
  dca_model,
  curve.names = "MSI model",
  xlab = "Threshold probability",
  ylab = "Net benefit",
  standardize = TRUE #-> for scaled NB: sNB= NB/prevalance (0.05)
  
)


# Extract the net benefit table
nb_df <- dca_model$derived.data
write.csv(nb_df,file.path(proj_dir, outcome_folder, model_folder,"ensemble_performance",paste0("DCA.csv")))

cond <- which(nb_df['cost.benefit.ratio'] == "1:4")
cond <- which(nb_df['sNB'] >= 0.45)
optimal_df <- nb_df[cond,]
optimal <- max(optimal_df$thresholds)

print(optimal)
new_th <- optimal
ensemble_pred_df[,"new_pred_class"] <- 0
cond <- ensemble_pred_df[,"mean_adj_prob_votedclass"] >= new_th
ensemble_pred_df[cond,"new_pred_class"] <- 1
new_res <- compute_classification_metrics(ensemble_pred_df, 
                                          truth_col = "True_y", 
                                          prob_col = "mean_adj_prob_votedclass",
                                          pred_col = "new_pred_class")
print(as.data.frame(new_res))


#Best AUCROC
#Compute ROC AUC
roc_obj <- roc(ensemble_pred_df[,"True_y"], ensemble_pred_df[,"mean_adj_prob_votedclass"])
auc_value <- auc(roc_obj)
print(auc_value)
plot(roc_obj, col="blue", main=paste("ROC Curve - AUC =", round(auc_value, 3)))

best <- coords(roc_obj, "best", ret=c("threshold", "sensitivity", "specificity"), best.method="youden")
print(best)

#PRAUC
actual <- ensemble_pred_df[,"True_y"]
predicted <- ensemble_pred_df[,"mean_adj_prob_votedclass"]
fg <- predicted[actual == 1]  # scores of positive class
bg <- predicted[actual == 0]  # scores of negative class
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = TRUE)
print(pr$auc.integral)
plot(pr)





# #TODO
# #compute_classification_metrics(ensemble_pred_df, truth_col = "True_y", prob_col = "mean_adj_prob_votedclass",pred_col = "majority_class")
# 
# df <- ensemble_pred_df
# colnames(df)[colnames(df) == 'mean_adj_prob_votedclass'] <- "p_hat"
# colnames(df)[colnames(df) == 'True_y'] <- "y_true"
# 
# # Put predictions into 10 bins (deciles)
# df$bin <- cut(df$p_hat,
#               breaks = quantile(df$p_hat, probs = seq(0, 1, 0.1), na.rm = TRUE),
#               include.lowest = TRUE)
# 
# # For each bin, compute mean predicted and observed risk
# calib <- aggregate(cbind(p_hat, y_true) ~ bin, data = df, FUN = mean)
# 
# ggplot(calib, aes(x = p_hat, y = y_true)) +
#   geom_point() +
#   geom_line() +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
#   labs(x = "Predicted probability",
#        y = "Observed event rate",
#        title = "Calibration plot") +
#   theme_minimal() +
#   ylim(0,0.3)
# 
# 
# # Make sure probabilities are strictly between 0 and 1
# eps   <- 1e-6
# p_adj <- pmin(pmax(df$p_hat, eps), 1 - eps)
# 
# # Log-odds of original predictions
# lp <- qlogis(p_adj)  # log(p/(1-p))
# y_true <- df$y_true
# 
# # Fit logistic recalibration model: logit(y) = alpha + beta * logit(p_hat)
# fit_cal <- glm(y_true ~ lp, family = binomial)
# 
# summary(fit_cal)  # look at intercept (alpha) and slope (beta)
# 
# alpha <- coef(fit_cal)[1]
# beta  <- coef(fit_cal)[2]
# 
# # Calibrated probabilities
# p_cal <- plogis(alpha + beta * lp)
# df$p_cal <- p_cal
# 
# # Put predictions into 10 bins (deciles)
# df$bin <- cut(df$p_cal,
#               breaks = quantile(df$p_cal, probs = seq(0, 1, 0.1), na.rm = TRUE),
#               include.lowest = TRUE)
# 
# # For each bin, compute mean predicted and observed risk
# calib <- aggregate(cbind(p_cal, y_true) ~ bin, data = df, FUN = mean)
# 
# ggplot(calib, aes(x = p_cal, y = y_true)) +
#   geom_point() +
#   geom_line() +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
#   labs(x = "Predicted probability",
#        y = "Observed event rate",
#        title = "Calibration plot") +
#   theme_minimal() +
#   ylim(0,0.3)

# library(rms)
# 
# df <- data.frame(y_true, p_hat)
# dd <- datadist(df); options(datadist = "dd")
# 
# fit <- lrm(y_true ~ p_hat, data = df, x = TRUE, y = TRUE)
# cal <- calibrate(fit, method = "boot", B = 200)  # bootstrap-corrected
# plot(cal, xlab = "Predicted probability", ylab = "Observed probability")

