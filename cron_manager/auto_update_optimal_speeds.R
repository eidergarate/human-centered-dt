library(logger)

setwd("/")

log_appender(appender_file("shared/speeds.log", max_lines=100, max_files=2L))

source(here::here("R", "main", "main_update_optimal_speeds.R"))
update_optimal_speeds()