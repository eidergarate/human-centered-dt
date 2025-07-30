library(dplyr)
library(magrittr)

add_optimal_speeds_production <- function(X0, optimal_speeds_path){
  
  optimal_speeds <- read.csv(optimal_speeds_path)
  
  if(!is.null(X0$recipe) & !is.na(X0$recipe)){
    
    if(!X0$recipe %in% optimal_speeds$recipe){
      EX1_setpoint <- sample(optimal_speeds$EX1_setpoint_speed, size = 1)
      EX2_setpoint <- sample(optimal_speeds$EX2_setpoint_speed, size = 1)
      EX3_setpoint <- sample(optimal_speeds$EX3_setpoint_speed, size = 1)
      EX4_setpoint <- sample(optimal_speeds$EX4_setpoint_speed, size = 1)
      EX5_setpoint <- sample(optimal_speeds$EX5_setpoint_speed, size = 1)
      
      EX1_slope <- sample(optimal_speeds$EX1_speed_slope, size = 1)
      EX2_slope <- sample(optimal_speeds$EX2_speed_slope, size = 1)
      EX3_slope <- sample(optimal_speeds$EX3_speed_slope, size = 1)
      EX4_slope <- sample(optimal_speeds$EX4_speed_slope, size = 1)
      EX5_slope <- sample(optimal_speeds$EX5_speed_slope, size = 1)
    }
    else{
      values <- optimal_speeds %>% filter(recipe == X0$recipe)
      
      EX1_setpoint <- values$EX1_setpoint_speed
      EX2_setpoint <- values$EX2_setpoint_speed
      EX3_setpoint <- values$EX3_setpoint_speed
      EX4_setpoint <- values$EX4_setpoint_speed
      EX5_setpoint <- values$EX5_setpoint_speed
      
      EX1_slope <- values$EX1_speed_slope
      EX2_slope <- values$EX2_speed_slope
      EX3_slope <- values$EX3_speed_slope
      EX4_slope <- values$EX4_speed_slope
      EX5_slope <- values$EX5_speed_slope
      
      
    }
    X0 %>% mutate(
      EX1_setpoint_speed = EX1_setpoint,
      EX2_setpoint_speed = EX2_setpoint,
      EX3_setpoint_speed = EX3_setpoint,
      EX4_setpoint_speed = EX4_setpoint,
      EX5_setpoint_speed = EX5_setpoint,
      
      EX1_speed_slope = EX1_slope,
      EX2_speed_slope = EX2_slope,
      EX3_speed_slope = EX3_slope,
      EX4_speed_slope = EX4_slope,
      EX5_speed_slope = EX5_slope,
    ) -> X
  }
  
  else{
    X0 %>% mutate(
      EX1_setpoint_speed = 0,
      EX2_setpoint_speed = 0,
      EX3_setpoint_speed = 0,
      EX4_setpoint_speed = 0,
      EX5_setpoint_speed = 0,
      
      EX1_speed_slope = 0,
      EX2_speed_slope = 0,
      EX3_speed_slope = 0,
      EX4_speed_slope = 0,
      EX5_speed_slope = 0,
    ) -> X
    
  }
  
  return(X)
}