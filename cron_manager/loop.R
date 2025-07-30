library(logger)
log_appender(appender_file("shared/loop.log", max_lines=100, max_files=2L))

loop_call <- function(x)
{
	while(TRUE){
		log_debug("LOOP!")
		tryCatch({
		  source(here::here("R","main","main_connect_to_influxDB_in_production.R"))
		  main_query_and_predict(con)},
				 error = function(e){
				   log_error(e)}
				 )

		Sys.sleep(x)
    }
}

loop_call(2)

