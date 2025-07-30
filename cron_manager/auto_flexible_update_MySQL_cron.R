library(logger)
log_appender(appender_file("shared/usage.log", max_lines=100, max_files=2L))

log_info("Usage address:    {usage_host}:{usage_port}/{usage_dbname}")
log_info("Usage credentials:{usage_user}:{usage_password}")

source(here::here("R", "main", "main_flexible_compute_indicators.R"))
main_flexible_compute_indicators()