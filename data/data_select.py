import pandas as pd
import os
import numpy as np
from shutil import copyfile

def data_select(source, destination, size):
    # Determine files that haven't been transfered
    source_files = [f for f in os.listdir(source) if f.endswith('.csv')]
    dest_files = [f for f in os.listdir(destination) if f.endswith('.csv')]
    diff_files = [ f for f in source_files if f not in dest_files ]
    
    # Select size amount of files to transfer
    if len(diff_files) > size:
        selected = np.random.choice(diff_files, size, replace=False)
    else:
        selected = diff_files
    
    # Transfer to destination
    for file in selected:
        copyfile( source + "/" + file, destination + "/" + file )
    

if __name__ == '__main__':
    source = "./data_use"
    destination = "./data_selected"
    
    data_select(source, destination, 32)
