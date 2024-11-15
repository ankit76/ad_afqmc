import os

def check_dir(dirname):                                                            
    if not os.path.exists(dirname):                                                
        os.makedirs(dirname)                                                       
                                                                                   
def read_real_array(h5, datname):                                                  
    return h5[datname][()]                                                         
                                                                                   
def read_complex_array(h5, datname):                                               
    real = datname + '_r'                                                          
    imag = datname + '_i'                                                          
    return h5[real][()] + 1.j * h5[imag][()]
