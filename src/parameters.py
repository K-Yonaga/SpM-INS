def get_parameters(data_name, zero_padding=False):
    if data_name == 'CsFeCl3(1.4GPa)':
        params = {
            'min_Q' : -0.55, 
            'max_Q' : -0.15, 
            'grid_Q' : 0.005,
            'min_E' : 1.0500000e-01, 
            'max_E' : 8.0500000e-01, 
            'grid_E' : 0.01,
            'min_I' : -20.0, 
            'max_I' : 0.05,
            'dQ' : 0.016,
            'dE' : 0.076,
            'zero_padding' : zero_padding}
        file_path = "./file/CsFeCl3_20180318_1p4GPa_KE_H0L0.iexy"
        
    elif data_name == 'CsFeCl3(0.0GPa)':
        params = {
            'min_Q' : -1.0, 
            'max_Q' : 0.0, 
            'grid_Q' : 0.01,
            'min_E' : 0.4, 
            'max_E' : 1.4875000e+00, 
            'grid_E' : 0.025,
            'min_I' : -200.0, 
            'max_I' : 200.0,
            'dQ' : 0.013,
            'dE' : 0.191,
            'zero_padding' : zero_padding}
        file_path = "./file/CsFeCl3_20160507_0p0GPa_KE_H0L0.iexy"
    
    elif data_name == 'Ba2NiTeO6(15meV)':
        params = {
            'min_Q' : 0.5, 
            'max_Q' : 2.5, 
            'grid_Q' : 0.03,
            'min_E' : 1.0, 
            'max_E' : 5.0, 
            'grid_E' : 0.1,
            'min_I' : 0.0, 
            'max_I' : 0.05,
            'dQ' : 0.0386,
            #'dE' : 0.0377,
            'dE' : 0.4789,
            'zero_padding' : zero_padding}
        file_path = "./file/Ba2NiTeO6_INS_15meV_2K.iexy"
    
    elif data_name == 'Ba2NiTeO6(7.5meV)':
        params = {
            'min_Q' : 0.5, 
            'max_Q' : 2.5, 
            'grid_Q' : 0.03,
            'min_E' : 1.0, 
            'max_E' : 5.0, 
            'grid_E' : 0.1,
            'min_I' : 0.0, 
            'max_I' : 1.0,
            'dQ' : 0.0386,
            'dE' : 0.0377,
            'zero_padding' : zero_padding}
        file_path = "./file/Ba2NiTeO6_INS_7p5meV_2K.iexy"
    
    elif data_name == 'NdFe3(BO3)4':
        params = {
            'min_Q' : -1.8, 
            'max_Q' : -1.2, 
            'grid_Q' : 0.02,
            'min_E' : 0.5, 
            'max_E' : 3.0, 
            'grid_E' : 0.05,
            'min_I' : 0.0, 
            'max_I' : 200.0,
            'dQ' : 0.037,
            'dE' : 0.30,
           'zero_padding' : zero_padding}
        file_path = "./file/5min.iexy"
    
    else:
        raise ValueError('Not supproted')
    
    return params, file_path