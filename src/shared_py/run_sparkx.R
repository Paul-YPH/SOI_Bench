args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript run_spark_worker.R <input_rds> <output_csv> <num_cores>")
}

input_rds_path <- args[1]
output_csv_path <- args[2]
n_cores <- as.integer(args[3])
n_top_genes <- as.integer(args[4])

suppressPackageStartupMessages({
  library(SPARK)
  library(Matrix)
})

cat(paste0("[R-Worker] Loading RDS: ", input_rds_path, "\n"))
spark <- readRDS(input_rds_path)

cat(paste0("[R-Worker] Running sparkx with ", n_cores, " cores...\n"))
sparkX <- sparkx(spark@counts, spark@location, numCores=n_cores, option="mixture", verbose=FALSE)

df_gene <- sparkX[["res_mtest"]][,c("combinedPval","adjustedPval")]
result <- df_gene[order(df_gene$combinedPval, decreasing = FALSE),]

if (nrow(result) > n_top_genes) {
    result <- result[1:n_top_genes, ]
}

write.csv(result, file = output_csv_path)
cat(paste0("[R-Worker] Saved SVG results to: ", output_csv_path, "\n"))