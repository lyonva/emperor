import pandas as pd
import os
import numpy as np
from shutil import copyfile, move

def data_move(source, destination):
    # Determine files
    source_files = [f for f in os.listdir(source) if f.endswith('.csv')]
    for file in source_files:
        move( os.path.join(source, file), os.path.join(destination, file) )

def data_select(source, destination, size, other=None):
    # Determine files that haven't been transfered
    source_files = [f for f in os.listdir(source) if f.endswith('.csv')]
    dest_files = [f for f in os.listdir(destination) if f.endswith('.csv')]
    diff_files = [ f for f in source_files if f not in dest_files ]
    # If historic directory is specified, remove from pool
    if other is not None:
        other_files = [f for f in os.listdir(other) if f.endswith('.csv')]
        diff_files = [ f for f in diff_files if f not in other_files ]
    
    # Select size amount of files to transfer
    if len(diff_files) > size:
        selected = np.random.choice(diff_files, size, replace=False)
    else:
        selected = diff_files
    
    # Transfer to destination
    for file in selected:
        
        copyfile( os.path.join(source, file), os.path.join(destination, file) )
    

if __name__ == "__main__":
    source = "./data_use"
    destination = "./data_selected"
    historic = "./data_archive"
    
    data_move(destination, historic) # Move already used data
    data_select(source, destination, 0, other=historic)
