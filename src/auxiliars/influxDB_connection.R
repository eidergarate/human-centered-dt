library(influxdbclient)

connect_to_influxDB <- function(influx_url, influx_token, influx_org){
  
  con <- InfluxDBClient$new(url = influx_url,
                     token = influx_token,
                     org = influx_org)
  
  return(con)
}

influxDB_query_production <- function(con, from_date, uc_id, influx_bucket, influx_measurement){
  
  date_filter_query <- build_date_filter_query(from_date)
  
  filter_signals <- get_uc_signals_production(uc_id)
  
  #All query statements are confidential
  
  meas_filter <- sprintf('---', influx_measurement)
  
  bucket_str <- sprintf('---', influx_bucket)
  
  pivoting_str <- sprintf('---')
  
  filling_str <- build_fill(uc_id)
  
  query_str <- paste0(bucket_str, date_filter_query, meas_filter, filter_signals, pivoting_str, filling_str)
  
  data_influx <- con$query(query_str)
  
  return(data_influx)
}


build_fill <- function(uc_id){
  if(uc_id == "UC2"){
    vars_list <- c("") #Data is confidential
  }
  
  filling_query_line <- sprintf(paste0('')) #Confidential query
  
  return(filling_query_line)
}



build_date_filter_query <- function(from_date){
  
  date_filter_query <- '' #Confidential query
  
  date_filter_query <-  sprintf('', from_date) #Confidential query
  
  return(date_filter_query)
  
}

build_date_filter_query_from_to <- function(from_date, to_date){
  
  date_filter_query <- '' #Confidential query
  
  date_filter_query <-  sprintf('', from_date, to_date) #Confidential query
  
  return(date_filter_query)
  
}

influxDB_query_production_from_to <- function(con, from_date, to_date, uc_id, influx_bucket, influx_measurement){
  
  date_filter_query <- build_date_filter_query_from_to(from_date, to_date)
  
  filter_signals <- get_uc_signals_production(uc_id)
  
  #Queries to continental data are confidential
  
  meas_filter <- sprintf('', influx_measurement)
  
  bucket_str <- sprintf('', influx_bucket)
  
  pivoting_str <- sprintf('')
  
  filling_str <- build_fill(uc_id)
  
  query_str <- paste0(bucket_str, date_filter_query, meas_filter, filter_signals, pivoting_str, filling_str)
  
  data_influx <- con$query(query_str)

  return(data_influx)
}


get_uc_signals_production <- function(uc_id = "UC2"){

  if(uc_id == "UC2"){
    vars_list <- c("") #Confidential
  }
  
  uc_signals_production <- sprintf(paste0('')) #Queries are confidential
  
  
  return(uc_signals_production)
}

