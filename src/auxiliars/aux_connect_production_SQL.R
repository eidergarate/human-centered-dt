library(DBI)

connect_to_production_pred_SQL <- function(host, port, db_name, user, password){
  
  drv <- RJDBC::JDBC(driverClass = "com.mysql.cj.jdbc.Driver", classPath = here::here("Drivers", "mysql-connector-java-8.0.30.jar"))
  
  con <- RJDBC::dbConnect(drv, paste0("jdbc:mysql://",host, ":", port, "/", db_name), user = user, password = password)
  
  return(con)
}

connect_to_production_usage_SQL <- function(host, port, db_name, user, password){
  
  drv <- RJDBC::JDBC(driverClass = "com.mysql.cj.jdbc.Driver", classPath = here::here("Drivers", "mysql-connector-java-8.0.30.jar"))
  
  con <- RJDBC::dbConnect(drv, paste0("jdbc:mysql://",host, ":", port, "/", db_name), user = user, password = password)
  
  return(con)
}

disconnect_from_production_SQL <- function(con){
  
  RJDBC::dbDisconnect(con)
  
}


connect_to_feedback_SQL <- function(host,
                                    port,
                                    user,
                                    password,
                                    db_name){
  
  
  drv <- RJDBC::JDBC(driverClass = "com.mysql.cj.jdbc.Driver", classPath = here::here("Drivers", "mysql-connector-java-8.0.30.jar"))
  
  con <- RJDBC::dbConnect(drv, paste0("jdbc:mysql://",host, ":", port, "/", db_name), user = user, password = password)
  
  return(con)
  
}