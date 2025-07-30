library(dplyr)
library(magrittr)
library(reticulate)
library(tibble)
source_python("src/auxiliars/aux_density_estimation.py")
source_python("src/auxiliars/aux_retrain_and_predict_M1.py")
source_python("src/auxiliars/aux_retrain_and_predict_M2.py")
source_python("src/auxiliars/unscale_data.py")

source(here::here("src", "auxiliars", "aux_density_categorical.R"))
source(here::here("src", "auxiliars", "get_interactions.R"))


get_n_samples_all_vars <- function(data_path, recipe_ID, n_samples, method, pct_density){
  
  data_all <- read.csv(data_path, stringsAsFactors = F)
  
  data_all %<>% filter(recipe == recipe_ID) %>% filter(--- == "---") #Confidential filtering
  
  if(method == "density"){
    
    X_num <- data_all %>% select(---) #Confidential selection
    
    
    X_num_samples_l <- get_samples(X_num, n_samples, save_scaler = T, scaler_path)
    
    X_num_samples <- X_num_samples_l[[2]]
    
    X_num_samples %<>% unscale_data(.)
    
  }
  
  else if(method == "historical"){
    
    data_all %<>% arrange(quality)
    
    X_num <- data_all %>% select(---) #Confidential selection
    
    
    X_num_samples <- X_num %>% na.omit() %>% head(n = n_samples)
    
    
  }
  
  else if(method == "combined"){
    
    n_density <- round(n_samples*pct_density)
    n_historical <- n_samples - n_density
    
   
    X_num <- data_all %>% select(---) #Confidential selection
    
    X_num_samples_l <- get_samples(X_num, n_density, save_scaler, scaler_path)
    
    X_num_samples_density <- X_num_samples_l[[2]]
    
    X_num_samples_density %<>% unscale_data(.)
  
    
    X_num$quality <- data_all$quality
    X_num_samples_hist <- X_num %>% arrange(desc(quality)) %>% select(-c(quality)) %>% na.omit() %>% head(n = n_historical)
    
    X_num_samples <- rbind(X_num_samples_density, X_num_samples_hist)
  }
  
  X_cat <- data_all %>% select(---) #Confidential
  
  range_matrix <- get_range_matrix(X_num)
  
  X_num_samples %<>% check_ranges(., range_matrix)
  
  X_cat_samples <- get_n_cat_samples(X_cat, n_samples)
  
  X_samples <- cbind(X_num_samples, X_cat_samples)
  
  X_samples %<>% get_interactions_production(.)
  
  X_samples %<>% check_not_using_EX(.)
  
  return(list("X_samples" = X_samples, "range_matrix" = range_matrix))
  
}


check_not_using_EX <- function(X_all){
  
  # Confidential
  
  return(X_all)
}


min_range <- function(var_vect){
  
  var_vect_min <- 0.9*min(var_vect, na.rm = T)
  var_vect_min <- if_else(var_vect_min < 0, 0, var_vect_min)
  
  return(var_vect_min)
}


max_range <- function(var_vect){
  
  var_vect_max <- 1.1*max(var_vect, na.rm = T)
  var_vect_max <- if_else(var_vect_max < 0, 0, var_vect_max)
  
  return(var_vect_max)
}


get_range_matrix <- function(X_num){
  
  min_values <- X_num %>% summarise_all(min_range)
  max_values <- X_num %>% summarise_all(max_range)
  
  range_matrix <- data.frame(matrix(0, nrow = 2, ncol = ncol(min_values)))
  colnames(range_matrix) <- colnames(X_num)
  range_matrix[1, ] <- min_values
  range_matrix[2, ] <- max_values
  
  return(range_matrix)
}


add_recipes_dummies <- function(X_all, recipe_ID){
  
  recipes <- readRDS(here::here("Data", "most_used_recipes.RDS"))
  
  recipes_mat <- matrix(0, nrow = nrow(X_all), ncol = length(recipes) + 1)
  
  cols_no_recipes <- ncol(X_all)
  
  X_all <- cbind(X_all, recipes_mat)
  
  colnames(X_all)[(cols_no_recipes + 1):ncol(X_all)] <- c(sprintf("recipe%s", recipes), "others")
  
  column_names <- colnames(X_all)
  
  if(any(recipe_ID == recipes)){
    
    col_j <- which(column_names == sprintf("recipe%s", recipe_ID))
    
    X_all[, col_j] <- rep(1, nrow(X_all))
    
  }
  else{
    
    X_all$others <- rep(1, nrow(X_all))
    
  }
  
  return(X_all)
}


objective_function <- function(X_all, recipe_ID){
  
  #deberíamos meter aqui interacciones etc y escalado
  
  #predict steadiness
  M1_model_path <- "M1_model.pkl"
  M1_scaler_path <- "M1_scaler.pkl"
  importances_path <- "M1_importances.csv"
  
  steadiness_pred <- scale_and_predict_M1(data_filtered = X_all, scaler_path = M1_scaler_path, model_path = M1_model_path, importances_path = importances_path)
  
  
  M2_model_path <- "M2_model.pkl"
  M2_scaler_path <- "M2_scaler.pkl"
  
  X_all <- add_recipes_dummies(X_all, recipe_ID)
  
  fitness <- exp(scale_and_predict_M2(data_filtered = X_all, scaler_path = M2_scaler_path, model_path = M2_model_path))
  
  ##############################################################################
  
  #                       CONSTRAINTS
  
  #1. READINESS
  
  for(individuo in 1:nrow(X_all)){
    
    if(steadiness_pred[individuo] != 0){
      
      fitness[individuo] <- 5*((steadiness_pred[individuo] + 2)^(steadiness_pred[individuo] + 2))*fitness[individuo]

    }
  }
  
  #2. FACTIBLE SETPOINTS RELATION
  for(individuo in 1:nrow(X_all)){
    if(steadiness_pred[individuo] == 0 & fitness[individuo] < 10000){
      max_setup_t <- get_setup_t_Extruders(X_all[individuo, ])

      ratio <- max_setup_t/fitness[individuo]
      if(ratio > 1.5){

        fitness[individuo] <- max_setup_t + 10
      }
    }
  }
  
  return(fitness)
}

get_setup_t_Extruders <- function(X_individuo){
  
  speed_setup_t <- c()
  
  #EX1
  if(X_individuo$--- == ...){ #Confidential variable
    EX1_t <- X_individuo$EX1_setpoint_speed/X_individuo$EX1_speed_slope
    speed_setup_t <- c(speed_setup_t, EX1_t)
  }
  
  #EX2

  EX2_t <- X_individuo$EX2_setpoint_speed/X_individuo$EX2_speed_slope
  speed_setup_t <- c(speed_setup_t, EX2_t)

  
  #EX3
  if(X_individuo$--- == ...){ #Confidential variable
    EX3_t <- X_individuo$EX3_setpoint_speed/X_individuo$EX3_speed_slope
    speed_setup_t <- c(speed_setup_t, EX3_t)
  }
  
  #EX4
  if(X_individuo$--- == ...){ #Confidential variable
    EX4_t <- X_individuo$EX4_setpoint_speed/X_individuo$EX4_speed_slope
    speed_setup_t <- c(speed_setup_t, EX4_t)
  }
  
  #EX5
  if(X_individuo$--- == ...){ #Confidential variable
    EX5_t <- X_individuo$EX5_setpoint_speed/X_individuo$EX5_speed_slope
    speed_setup_t <- c(speed_setup_t, EX5_t)
  }
  
  setup_t <- max(speed_setup_t)
  
  return(setup_t)
  
}

check_speed_constraints <- function(unscaled_data){
  
 #Confidential
  
  return(speed_constraints)
  
}


check_temp_constraints <- function(unscaled_data){
  
  #Confidential
  return(temperature_constraints)
}


check_press_constraints <- function(unscaled_data){
  
  #Confidential
  
  return(pressure_constraints)
}


check_ranges <- function(population, range_matrix){
  
  lower_bound <-  range_matrix[1,]
  upper_bound <- range_matrix[2,]
  
  for(j in 1:ncol(population)){
    lower_oor <- which(population[,j] < lower_bound[, j])
    if(length(lower_oor) != 0){
      population[lower_oor, j] <- lower_bound[, j]
    }
    
    upper_oor <- which(population[,j] > upper_bound[, j])
    if(length(upper_oor) != 0){
      population[upper_oor, j] <- upper_bound[, j]
    }
    
  }
  
  
  return(population)
}


unscale_data_and_unify <- function(X_all){
  
  X_num_no_interactions <- X_all %>% select(---) #Confidential
  
  X_num_no_int_unscaled <- unscale_data(X_num = X_num_no_interactions)
  
  X_cat <- X_all %>% select(---) #Confidential
  
  X_num_cat <- cbind(X_num_no_int_unscaled, X_cat)
  
  X_num_cat %<>% get_interactions_production(.)
  
  return(X_num_cat)
}


DE_simple <- function(data_path, recipe_ID, n_pop, max_iter, F_parameter, Cr_parameter, method, mutation = "DE/rand/1", pct_density = 0){
  
  time0 <- Sys.time()
  n <- 64
  ini_pop <- get_n_samples_all_vars(data_path, recipe_ID, n_pop, method, pct_density)

  pop <- ini_pop$X_samples %>% select(-c(id))
  rownames(pop) <- 1:n_pop
  range_matrix <- ini_pop$range_matrix
  
  pop$pop_idx <- 1:n_pop
  
  time0 <- Sys.time()
  f_pop <- objective_function(pop %>% select(-c(pop_idx)), recipe_ID)
  f_pop <- data.frame("f_pop" = f_pop, "pop_idx" = pop$pop_idx)
  
  all_solutions <- pop[, c(1:64, 119)]
  all_solutions$iter <- rep(0, n_pop)
  all_solutions$fitness <- f_pop$f_pop
  
  
  
  time0 <- Sys.time()
  x_best_i <- pop[which.min(f_pop$f_pop), ]
  distancia_mn_new <- mean(sqrt(rowSums(pop[, 1:64] - outer(rep(1, n_pop), t(x_best_i[, 1:64]))[, , 1])^2))
  distancia_mn_vect <- rep(0, max_iter + 1)
  distancia_mn_vect[1] <- distancia_mn_new

  time_bucle <- Sys.time()
  for(iter in 1:max_iter){
    
    #Mutation matrix
    V_G <- data.frame(matrix(0, nrow = n_pop, ncol = 64))
    colnames(V_G) <- colnames(pop)[1:64]
    
    #Crossover matrix
    U_G <- data.frame(matrix(0, nrow = n_pop, ncol = 64))
    colnames(U_G) <- colnames(pop)[1:64]
    
    time_mut <- Sys.time()
    
    if(mutation == "DE/rand/1"){
      
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){

        r_indexes <- sample(setdiff(1:n_pop, particle), 3)
        
        X_r1 <- pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]
        X_r3 <-  pop %>% filter(pop_idx == r_indexes[3]); X_r3 <- X_r3[,1:64]
        
        V_G_particle <- X_r1 + F_parameter*(X_r2 - X_r3)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle 
        return(V_G_particle)
        
      }, n_pop, pop, F_parameter) %>% bind_rows()
      
    }
    
    else if(mutation == "DE/rand/2"){
      
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){
        
        r_indexes <- sample(setdiff(1:n_pop, particle), 5)

        X_r1 <-  pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]
        X_r3 <-  pop %>% filter(pop_idx == r_indexes[3]); X_r3 <- X_r3[,1:64]
        X_r4 <-  pop %>% filter(pop_idx == r_indexes[4]); X_r4 <- X_r4[,1:64]
        X_r5 <-  pop %>% filter(pop_idx == r_indexes[5]); X_r5 <- X_r5[,1:64]

        V_G_particle <- X_r1 + F_parameter*(X_r2 - X_r3) + F_parameter*(X_r4 - X_r5)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle
        return(V_G_particle)
        
      }, n_pop, pop, F_parameter) %>% bind_rows()
      
    }
    
    else if(mutation == "DE/best/1"){
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){
        
        r_indexes <- sample(setdiff(1:n_pop, particle), 2)
        
        X_best_pop_idx <- f_pop[which.min(f_pop$f_pop), ]$pop_idx
        
        X_best <- pop %>% filter(pop_idx == X_best_pop_idx); X_best <- X_best[,1:64]
        X_r1 <-  pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]

        V_G_particle <- X_best + F_parameter*(X_r1 - X_r2)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle
        return(V_G_particle)
        
      }, n_pop, pop, F_parameter) %>% bind_rows()
      
    }
    
    else if(mutation == "DE/best/2"){
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){
        
        r_indexes <- sample(setdiff(1:n_pop, particle), 4)

        X_best_pop_idx <- f_pop[which.min(f_pop$f_pop), ]$pop_idx
        
        X_best <- pop %>% filter(pop_idx == X_best_pop_idx); X_best <- X_best[,1:64]
        X_r1 <-  pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]
        X_r3 <-  pop %>% filter(pop_idx == r_indexes[3]); X_r3 <- X_r3[,1:64]
        X_r4 <-  pop %>% filter(pop_idx == r_indexes[4]); X_r4 <- X_r4[,1:64]

        V_G_particle <- X_best + F_parameter*(X_r1 - X_r2) + F_parameter*(X_r3 - X_r4)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle
        return(V_G_particle)
        
      }, n_pop, pop, F_parameter) %>% bind_rows()
      
    }
    
    else if(mutation == "DE/current-to-rand/1"){
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){

        r_indexes <- sample(setdiff(1:n_pop, particle), 3)

        X_r1 <-  pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]
        X_r3 <-  pop %>% filter(pop_idx == r_indexes[3]); X_r3 <- X_r3[,1:64]
        X_particle <- pop %>% filter(pop_idx == particle); X_particle <- X_particle[,1:64]

        V_G_particle <- X_particle + F_parameter*(X_r1 - X_particle) + F_parameter*(X_r2 - X_r3)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle
        return(V_G_particle)

      }, n_pop, pop, F_parameter) %>% bind_rows()

    }
    
    else if(mutation == "DE/current-to-rand/2"){
      V_G <- lapply(1:n_pop,  FUN = function(particle, n_pop, pop, F_parameter){

        r_indexes <- sample(setdiff(1:n_pop, particle), 2)

        X_best_pop_idx <- f_pop[which.min(f_pop$f_pop), ]$pop_idx
        
        X_best <- pop %>% filter(pop_idx == X_best_pop_idx); X_best <- X_best[,1:64]
        X_r1 <-  pop %>% filter(pop_idx == r_indexes[1]); X_r1 <- X_r1[,1:64]
        X_r2 <-  pop %>% filter(pop_idx == r_indexes[2]); X_r2 <- X_r2[,1:64]
        X_particle <- pop %>% filter(pop_idx == particle); X_particle <- X_particle[,1:64]

        V_G_particle <- X_particle + F_parameter*(X_best - X_particle) + F_parameter*(X_r1 - X_r2)
        colnames(V_G_particle) <- colnames(pop)[1:64]
        V_G_particle$pop_idx <- particle
        return(V_G_particle)
        
      }, n_pop, pop, F_parameter) %>% bind_rows()
      
    }
    
    time_cross <- Sys.time()
    rand_values <- runif(n_pop)
    U_G <- lapply(1:n_pop, FUN = function(particle, Cr_parameter, rand_values, V_G, pop){

      if(rand_values[particle] <= Cr_parameter){
        result <- V_G %>% filter(pop_idx == particle); result <- result[,1:64]
      }else{
        result <-  pop %>% filter(pop_idx == particle); result <- result[,1:64]
      }
      
      result$pop_idx <- particle
      return(result)},
      Cr_parameter, rand_values, V_G, pop) %>% bind_rows()
    
    
    U_G %<>% remove_rownames %>% column_to_rownames(var = "pop_idx")
    
    U_G %<>% check_ranges(., range_matrix)
    
    U_G <- cbind(U_G, pop[ ,65:68])
    
    U_G %<>% get_interactions_production(.)
    
    U_G %<>% check_not_using_EX(.)
    
    U_G %<>% rownames_to_column(var = "pop_idx")
    U_G$pop_idx <- as.numeric(U_G$pop_idx)
    
    # "Evaluation"
    time_eval <- Sys.time()
    
    f_ug <- objective_function(U_G %>% select(-c(pop_idx)), recipe_ID)
    f_ug <- data.frame("f_ug" = f_ug, "pop_idx" = U_G$pop_idx)

    # "SELECTION"
    
    f_pop_ug <- merge(f_pop, f_ug, by = "pop_idx")
    f_pop_ug %<>% mutate(comparative = if_else(f_ug < f_pop, T, F), f_pop = if_else(comparative, f_ug, f_pop)) 
    
    f_pop <- f_pop_ug %>% select(f_pop, pop_idx)
    
    change_pop_idxs <- f_pop_ug[which(f_pop_ug$comparative), ]$pop_idx
    
    for(pop_idx in change_pop_idxs){
      
      pop[which(pop$pop_idx == pop_idx), c(1:64, 119)] <- U_G[which(U_G$pop_idx == pop_idx), c(2:65, 1)]
      
    }
      

    x_best_i <- pop[which.min(f_pop$f_pop), ]
    distancia_mn_new <- mean(sqrt(rowSums(pop[, 1:64] - outer(rep(1, n_pop), t(x_best_i[, 1:64]))[, , 1])^2))
    distancia_mn_vect[(iter + 1)] <- distancia_mn_new
    
    time_sol <- Sys.time()
    iter_solution <- pop[, 1:64]
    iter_solution$iter <- iter
    iter_solution$pop_idx <- pop$pop_idx
    iter_solution <- merge(iter_solution, f_pop, by = "pop_idx")
    iter_solution <- cbind(iter_solution[, 2:65], iter_solution[, 1], iter_solution[, 66:67])
    colnames(iter_solution)[67] <- "fitness"
    colnames(iter_solution)[65] <- "pop_idx"
    
    all_solutions <- rbind(all_solutions, iter_solution)
    

  } 
  
  return(list("fitness" = f_pop, "last_population" = pop, "all_solutions" = all_solutions, "dispersion" = distancia_mn_vect))
}
