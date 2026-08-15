# ============================================================
# SCORE Dissertation — Run hierarchical Firth regression on the
# real regression_dataset.csv
#
# Run test_synthetic_data.R successfully FIRST. This script only
# reads/writes on the secure drive, per governance.
# ============================================================

source("firth_regression_functions.R")

DATA_PATH <- "C:/Users/mattn/Documents/Dissertation R/regression_dataset.csv"
OUT_DIR   <- "C:/Users/mattn/Documents/Dissertation R/regression_outputs"

if (!dir.exists(OUT_DIR)) dir.create(OUT_DIR, recursive = TRUE)

cat("R version:", R.version.string, "\n")
cat("logistf version:", as.character(packageVersion("logistf")), "\n\n")

df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

cat("=== Real data checks ===\n")
cat("N rows read:", nrow(df), "\n")
if (nrow(df) != 298) {
  stop("N is not 298 -- STOP. Do not proceed until this is understood ",
       "(handoff doc: 'if you re-run the assembly script and these ",
       "numbers move, don't proceed until that's understood').")
}
if (!"post_covid" %in% names(df)) {
  stop("post_covid column not found -- check it was joined in correctly.")
}
cat("post_covid split:", table(df$post_covid), "\n")
cat("attended prevalence:", round(mean(df$attended), 4),
    paste0("(", sum(df$attended), "/", nrow(df), ")"), "\n\n")

results <- run_both_label_sources(df, expected_n = 298)
cat("Assertions passed: N and formula-consistency checks OK\n\n")

for (src in c("gold", "pred")) {
  cat("=====", src, "=====\n")
  print(summary(results[[src]]$fit_full))
  cat("\nBlock LRT:\n")
  print(results[[src]]$block_lrt)
  cat("\n")
}

# Assemble and save the table-ready output
tables <- lapply(c("gold", "pred"), function(src) build_regression_table(results[[src]], df))
names(tables) <- c("gold", "pred")

write.csv(tables$gold$body,       file.path(OUT_DIR, "regression_table_gold_body.csv"), row.names = FALSE)
write.csv(tables$gold$fit_stats,  file.path(OUT_DIR, "regression_table_gold_fitstats.csv"), row.names = FALSE)
write.csv(tables$pred$body,       file.path(OUT_DIR, "regression_table_pred_body.csv"), row.names = FALSE)
write.csv(tables$pred$fit_stats,  file.path(OUT_DIR, "regression_table_pred_fitstats.csv"), row.names = FALSE)

saveRDS(results, file.path(OUT_DIR, "regression_results_full.rds"))

cat("Saved table CSVs and full results .rds to:\n", OUT_DIR, "\n")
