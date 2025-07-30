import pandas as pd
import numpy as np
import re
from datetime import datetime


def detect_time(log_line):
    
    date_time = re.search(r'\[(.*?)\]', log_line).group(1)
    time = date_time[11:19]
  
    return date_time, time

def detect_text(log_line):
  
    message = log_line[27:]
  
    return message
  
def get_time_difference(time0_str, timef_str):
  
    time0 = datetime.strptime(time0_str,"%H:%M:%S")
    timef = datetime.strptime(timef_str,"%H:%M:%S")

    difference = timef - time0
    difference = difference.total_seconds()
  
    return(difference)

def readlog_files(log_path):
  
  
    colum_names = ['time_stamp', 'influx_query', 'data_cleaning_processing',
    'optimal_speeds', 'prediction_time', 'suggestions', 'update_mysql']
    
    computational_cost = pd.DataFrame(columns = colum_names)
  
    timings = 0
    influx = 0 
    data_processing = 0
    speeds =  0
    predict = 0
    suggestions = 0
    loading = 0
  
    pred = False
  
    ini = False
    ini_idx = -1

    with open(log_path, 'r') as file:
        
        lines = [line.strip() for line in file]
        
    for index, line in enumerate(lines, start = 0): 
        date_time, time = detect_time(line)
        message = detect_text(line)
      
        if(message == 'Querying production SQL to get timings for influx query'): 
            ini = True
            ini_idx = ini_idx + 1
        
        
        if(ini and index <= len(lines) - 2):
            if(message == "Querying production SQL to get timings for influx query"):
                timings = time
            
                next_message_compute = lines[(index + 1)]
                next_date_time, next_time = detect_time(next_message_compute)
                difference = get_time_difference(timings, next_time)
                computational_cost.loc[ini_idx, 'time_stamp'] = difference
            
            elif(message == "Influx querying"):
                influx = time
        
                next_message_compute = lines[(index + 1)]
                next_date_time, next_time = detect_time(next_message_compute)
                difference = get_time_difference(influx, next_time)
                computational_cost.loc[ini_idx, 'influx_query'] = difference
            
            elif(message == "Prediction is initialized."):
                pred = True
            
            if(pred and index <= len(lines) - 1):
            
                if(message == "Correcting last extrusion's data"):
                    data_processing = time
                    
                    if(index <= len(lines) - 3):
                        next_message_compute = lines[(index + 2)]
                        next_date_time, next_time = detect_time(next_message_compute)
                        difference = get_time_difference(data_processing, next_time)
                        computational_cost.loc[ini_idx, 'data_cleaning_processing'] = difference
              
                elif(message == "Adding M2's output"):
                    speeds = time
              
                    next_message_compute = lines[(index + 1)]
                    next_date_time, next_time = detect_time(next_message_compute)
                    difference = get_time_difference(speeds, next_time)
                    computational_cost.loc[ini_idx, 'optimal_speeds'] = difference
              
                elif(message == "Loading model and predicting"):
                    predict = time
              
                    next_message_compute = lines[(index + 1)]
                    next_date_time, next_time = detect_time(next_message_compute)
                    difference = get_time_difference(predict, next_time)
                    computational_cost.loc[ini_idx, 'prediction_time'] = difference
              
                elif(message == "Adding suggestions"):
                    suggestions = time
               
                    next_message_compute = lines[(index + 1)]
                    next_date_time, next_time = detect_time(next_message_compute)
                    difference = get_time_difference(suggestions, next_time)
                    computational_cost.loc[ini_idx, 'suggestions'] = difference
              
                elif(message == "Updating Production SQL with the readiness prediction."):
                    loading = time
              
                    next_message_compute = lines[(index + 1)]
                    next_date_time, next_time = detect_time(next_message_compute)
                    difference = get_time_difference(loading, next_time)
                    computational_cost.loc[ini_idx, 'update_mysql'] = difference
             
           
            if(message == "Disconnecting from production SQL."):
                ini = False
                pred = False
                
            
    return computational_cost

def get_computational_cost(log_paths):
    
    computational_cost = readlog_files(log_paths[0])
    
    for log in range(1, len(log_paths)):
      
        log_path2 = log_paths[log]
        computational_cost_log2 = readlog_files(log_path2)
        computational_cost = pd.concat([computational_cost, computational_cost_log2])
  
    return computational_cost


          
log_paths = ['---'] #Confidential
  
  
computational_cost =  get_computational_cost(log_paths) 
  
computational_cost.to_csv('name.csv')  
  
  
  
