# ============================================================
# SCORE Dissertation — Hierarchical Firth Logistic Regression
# Core functions
#
# Block 1: risk_group + n_sentences + post_covid
# Block 2: adds the seven emotion predictors
# Confirmatory test: block-level LRT via anova.logistf() (penalty-consistent)
# Run once per label source (gold_/pred_) via the SAME function.
# ============================================================

suppressMessages(library(logistf))

EMOTIONS <- c("anxiety", "sadness", "overwhelm", "amotivation",
              "shame", "hopelessness", "loneliness")
BLOCK1_VARS <- c("risk_group", "n_sentences", "post_covid")

# ---- Manual VIF (auxiliary-regression definition) ----------
# VIF is a property of the predictor design matrix, not of the outcome
# model, so it doesn't need car::vif() or a glm() stand-in at all --
# computing it directly via auxiliary OLS regressions sidesteps any
# question of whether car::vif() supports a logistf object.
compute_vif_manual <- function(design_matrix) {
  vars <- colnames(design_matrix)
  vifs <- sapply(vars, function(v) {
    y <- design_matrix[, v]
    X <- design_matrix[, vars != v, drop = FALSE]
    r2 <- summary(lm(y ~ X))$r.squared
    if (r2 >= 1) return(Inf)
    1 / (1 - r2)
  })
  setNames(vifs, vars)
}

# ---- Core fitting function -----------------------------------
# prefix: "gold_" or "pred_" -- selects which set of emotion columns
# to use. This is the ONLY thing that differs between the two calls,
# so gold and pred runs cannot drift apart in specification.
fit_hierarchical_firth <- function(data, prefix = c("gold_", "pred_")) {
  prefix <- match.arg(prefix)
  emo_cols <- paste0(prefix, EMOTIONS)

  stopifnot(
    "emotion columns missing for this prefix" = all(emo_cols %in% names(data)),
    "block 1 covariates missing" = all(BLOCK1_VARS %in% names(data)),
    "attended column missing" = "attended" %in% names(data)
  )

  data$risk_group <- factor(data$risk_group, levels = c("low", "high"))
  # post_covid assumed already 0/1 numeric (0 = pre, 1 = post), per the
  # coding agreed and applied in the Excel column.

  f_block1 <- as.formula(paste("attended ~", paste(BLOCK1_VARS, collapse = " + ")))
  f_full   <- as.formula(paste("attended ~", paste(c(BLOCK1_VARS, emo_cols), collapse = " + ")))

  fit_block1 <- logistf(f_block1, data = data)
  fit_full   <- logistf(f_full,   data = data)

  # Block-level LRT: anova.logistf() re-derives the null penalized
  # likelihood within the SAME penalty scope as fit_full, so this is
  # NOT the same as subtracting the two models' own $loglik['null']
  # values (each of those is against an intercept-only null under that
  # model's own, differently-scoped, penalty).
  block_lrt <- anova(fit_full, fit_block1)

  # McFadden's pseudo-R^2 for the full model, against ITS OWN intercept-only
  # null (this is the conventional McFadden definition, and is a single-
  # model quantity fit_full already carries internally) -- separate from
  # and not a substitute for the block_lrt comparison above.
  pseudo_r2_full <- 1 - fit_full$loglik["full"] / fit_full$loglik["null"]
  names(pseudo_r2_full) <- NULL

  # VIF on the full block-1+2 design matrix (excludes intercept)
  design <- model.matrix(f_full, data = data)[, -1, drop = FALSE]
  vif_vals <- compute_vif_manual(design)

  or_table <- data.frame(
    term    = names(coef(fit_full)),
    OR      = exp(unname(coef(fit_full))),
    CI_low  = exp(fit_full$ci.lower),
    CI_high = exp(fit_full$ci.upper),
    p       = fit_full$prob,
    row.names = NULL
  )

  list(
    prefix        = prefix,
    n             = nrow(data),
    fit_block1    = fit_block1,
    fit_full      = fit_full,
    block_lrt     = block_lrt,
    # logistf objects have no logLik()/AIC() method (confirmed empirically --
    # stats::AIC()'s default method depends on logLik(), which logistf
    # doesn't provide). extractAIC.logistf() IS defined and is the correct
    # route; it returns c(edf, AIC), so we take the second element.
    aic_block1    = unname(extractAIC(fit_block1)[2]),
    aic_full      = unname(extractAIC(fit_full)[2]),
    pseudo_r2_full = pseudo_r2_full,
    vif           = vif_vals,
    or_table      = or_table,
    formula_block1 = f_block1,
    formula_full   = f_full
  )
}

# ---- Wrapper: run both label sources + safety assertions -----
run_both_label_sources <- function(data, expected_n = 298) {
  stopifnot("N has drifted from the agreed 298" = nrow(data) == expected_n)

  res_gold <- fit_hierarchical_firth(data, "gold_")
  res_pred <- fit_hierarchical_firth(data, "pred_")

  # Confirm the two runs used identical specifications apart from the
  # column prefix -- guards against the two runs silently drifting apart.
  strip_prefix <- function(f) gsub("gold_|pred_", "", deparse(f))
  stopifnot(
    "gold/pred block1 formulas differ beyond prefix" =
      identical(strip_prefix(res_gold$formula_block1), strip_prefix(res_pred$formula_block1)),
    "gold/pred full formulas differ beyond prefix" =
      identical(strip_prefix(res_gold$formula_full), strip_prefix(res_pred$formula_full))
  )

  list(gold = res_gold, pred = res_pred)
}

# ---- Table assembly: matches the regression table structure --
# Body: term, prevalence (%), OR, 95% CI, p
# Footer (returned separately as $fit_stats): block1 AIC, full AIC,
# McFadden R2, block LRT chisq/df/p, per-term VIF
build_regression_table <- function(result, data) {
  emo_cols <- paste0(result$prefix, EMOTIONS)
  prevalence <- colMeans(data[emo_cols]) * 100
  names(prevalence) <- sub(result$prefix, "", names(prevalence))

  body <- result$or_table
  body$prevalence_pct <- NA_real_
  for (e in names(prevalence)) {
    body$prevalence_pct[body$term == paste0(result$prefix, e)] <- prevalence[[e]]
  }
  body$vif <- NA_real_
  matched <- match(body$term, names(result$vif))
  body$vif[!is.na(matched)] <- result$vif[matched[!is.na(matched)]]

  fit_stats <- data.frame(
    label_source   = result$prefix,
    n               = result$n,
    aic_block1      = result$aic_block1,
    aic_full        = result$aic_full,
    pseudo_r2_full  = result$pseudo_r2_full,
    lrt_chisq       = unname(result$block_lrt$chisq),
    lrt_df          = result$block_lrt$df,
    lrt_p           = unname(result$block_lrt$pval)
  )

  list(body = body, fit_stats = fit_stats)
}
