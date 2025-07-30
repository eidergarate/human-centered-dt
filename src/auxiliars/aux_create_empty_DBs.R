source(here::here("src", "auxiliars", "aux_connect_historicDB.R"))
source(here::here("src", "auxiliars", "aux_connect_production_SQL.R"))
library(RMariaDB)
library(DBI)
#creamos el empty historic

create_empty_historicDB <- function(host, port,
                                    user, password,
                                    historic_DB_name,
                                    preprocessed_table_name,
                                    errors_table_name){
  
  #Preprocessed table empty
  ExtrucExtract::create_empty_DB(host, port,
                                 user, password,
                                 DB_name,
                                 table_name, quality_limit = 0.01)
  
  
  #errors table empty
  con <- connect_to_historicDB()
  
  errors_empty_df <- data.frame(unique_ID, EX_DS_Hot_prof_status, M1_pred, M1_error, model_version, stringsAsFactors)
  
  
  
  if(!dbExistsTable(con, errors_table_name)){
    
    RMariaDB::dbWriteTable(con, errors_table_name, errors_empty_df)
    
  }else{
    
    
    RMariaDB::dbAppendTable(con, errors_table_name, errors_empty_df)
    
  }
  
  disconnect_from_historicDB(con)
}

create_empty_production_SQL <- function(db_name,
                                        prediction_table,
                                        usage_table){
  
  con <- connect_to_production_SQL()
  

  RJDBC::dbSendUpdate(con, paste0("CREATE TABLE `", prediction_table, "` (`Date` DATETIME NOT NULL, `Pred` VARCHAR(255) NOT NULL, `Suggestions` VARCHAR(255) NOT NULL);"))
  
  
  #crear tabla de usage

  RJDBC::dbSendUpdate(con, paste0("CREATE TABLE `",  usage_table, "` (`Date` DATETIME NOT NULL, `Usage` BOOLEAN DEFAULT FALSE);"))
  
  
  
  disconnect_from_production_SQL(con)
}


create_empty_feedback_SQL <- function(db_name,
                                      feedback_table){
  
  
  con <- connect_to_feedback_SQL()
  
  #crete Table
  
  query <- sprintf("CREATE TABLE %s (id INT NOT NULL, PRIMARY KEY (id), Date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, Valid BOOLEAN DEFAULT FALSE, Reason VARCHAR(255) NOT NULL)", feedback_table)

  RJDBC::dbSendUpdate(con, query)
  
  disconnect_from_production_SQL(con)
}










