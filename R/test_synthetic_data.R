# ============================================================
# Synthetic-data test of firth_regression_functions.R
# Matches the real regression_dataset.csv column structure and N.
# Run this BEFORE pointing run_regression_real.R at the real file.
# ============================================================

source("firth_regression_functions.R")
set.seed(42)

N <- 298

make_synthetic_data <- function(n = N) {
  risk_group <- sample(c("low", "high"), n, replace = TRUE, prob = c(0.6, 0.4))
  n_sentences <- pmax(1, rpois(n, lambda = 4))
  # match the real pre/post split you reported: 73 pre, 225 post
  post_covid <- c(rep(0, 73), rep(1, n - 73))
  post_covid <- sample(post_covid)  # shuffle order

  # emotion prevalences: mix of common and rare, to stress-test the
  # rare classes (hopelessness, shame, loneliness, amotivation) the
  # same way the real corpus does
  prevalence <- c(anxiety = 0.35, sadness = 0.30, overwhelm = 0.25,
                   amotivation = 0.08, shame = 0.06, hopelessness = 0.07,
                   loneliness = 0.10)

  make_emotion_set <- function(prefix) {
    cols <- lapply(EMOTIONS, function(e) rbinom(n, 1, prevalence[e]))
    names(cols) <- paste0(prefix, EMOTIONS)
    as.data.frame(cols)
  }

  gold_df <- make_emotion_set("gold_")
  # pred_ correlated with gold_ but noisier (simulates classifier error)
  pred_df <- as.data.frame(lapply(gold_df, function(col) {
    flip <- rbinom(n, 1, 0.15) == 1
    ifelse(flip, 1 - col, col)
  }))
  names(pred_df) <- paste0("pred_", EMOTIONS)

  attended <- rbinom(n, 1, 0.705)

  data.frame(
    Text_ID = seq_len(n),
    gold_df,
    pred_df,
    n_sentences = n_sentences,
    attended = attended,
    risk_group = risk_group,
    post_covid = post_covid
  )
}

df <- make_synthetic_data()

cat("=== Synthetic data summary ===\n")
cat("N:", nrow(df), "\n")
cat("post_covid split:", table(df$post_covid), "\n")
cat("attended prevalence:", mean(df$attended), "\n\n")

cat("=== Running run_both_label_sources() ===\n")
results <- run_both_label_sources(df, expected_n = N)
cat("Assertions passed: N and formula-consistency checks OK\n\n")

for (src in c("gold", "pred")) {
  r <- results[[src]]
  cat("---", src, "---\n")
  cat("Block1 formula:", deparse(r$formula_block1), "\n")
  cat("Full formula:  ", deparse(r$formula_full), "\n")
  cat("AIC block1:", round(r$aic_block1, 2), " AIC full:", round(r$aic_full, 2), "\n")
  cat("McFadden pseudo-R2 (full):", round(r$pseudo_r2_full, 4), "\n")
  cat("Block LRT chisq:", round(r$block_lrt$chisq, 3),
      " df:", r$block_lrt$df, " p:", signif(r$block_lrt$pval, 4), "\n")
  cat("VIF range:", round(range(r$vif), 2), "\n")
  cat("OR table (first 3 rows):\n")
  print(head(r$or_table, 3))
  cat("\n")
}

cat("=== Testing build_regression_table() ===\n")
for (src in c("gold", "pred")) {
  tbl <- build_regression_table(results[[src]], df)
  cat("---", src, "body ---\n")
  print(tbl$body)
  cat("---", src, "fit_stats ---\n")
  print(tbl$fit_stats)
  cat("\n")
}

cat("=== Test complete: no errors ===\n")
