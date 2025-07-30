library(DBI)
library(dplyr)
library(RMariaDB)

connect_to_historicDB <- function(host, port, user, password, db_name){
  
  con <- RMariaDB::dbConnect(RMariaDB::MariaDB(), host = host, port = port, user = user, password = password, dbname = db_name)
  
  return(con)
}

disconnect_from_historicDB <- function(con){
  
  RMariaDB::dbDisconnect(con)
}

query_historicalDB_preprocessed_all <- function(host, port, user, password, db_name, table_name){
  
  con <- connect_to_historicDB(host, port, user, password, db_name)
  
  tbl(con, table_name) %>% collect() -> preprocessed_data
  
  disconnect_from_historicDB(con)
  
  return(preprocessed_data)
}