import numpy as np
import json

class DataNormalizer:
    def __init__(self, stat_dict:dict|str, data_dict:dict|str, method=None, level=0.98):
        
        if isinstance(stat_dict,str):
            with open(stat_dict) as json_file:
                self.stat_dict = json.load(json_file)

       
        if isinstance(data_dict,str):
            with open(data_dict) as json_file:
                data_dict = json.load(json_file)

        if method is not None:
            variables = np.unique(np.array(data_dict['train']['source']['variables'] 
                                           + data_dict['train']['target']['variables']))
            self.var_norm_dict = {}
            for variable in variables:
                self.var_norm_dict[variable] = {"method": method,
                                                "level": level}
            
        elif 'normalization' in data_dict.keys():
            self.var_norm_dict = data_dict['normalization']          
     
    
    def normalize(self, data, var):
        stats = self.stat_dict[var]
        norm = self.var_norm_dict[var]

        q_low_  = np.min([norm['level'], 1-norm['level']])
        q_high_ = np.max([norm['level'], 1-norm['level']])

        if norm['method']=='quantile':
            q_low = stats['quantiles'][str(q_low_)]
            q_high = stats['quantiles'][str(q_high_)]

            return (data - q_low)/(q_high - q_low)
                
        elif norm['method']=='quantile_abs':
            q_high = stats['quantiles'][str(q_high_)]
            data = data / q_high
            return data

        elif norm['method']=='min_max':
            return (data - stats['min'])/(stats['max'] - stats['min'])
        
        elif norm['method']=='normal':
            return (data - stats['mean'])/(stats['std'])
        
    def renormalize(self, data, var):
        stats = self.var_norm_dict[var]
        norm = self.var_norm_dict[var]

        if norm['method']=='quantile':
            q_low = stats['quantiles'][str(1-float(norm['level']))]
            q_high = stats['quantiles'][str((norm['level']))]

            return (data)*(q_high - q_low) + q_low
                
        elif norm['method']=='quantile_abs':
            q_high = stats['quantiles'][str((norm['level']))]
            data = data * q_high

        elif norm['method']=='min_max':
            return (data) * (stats['max'] - stats['min']) + stats['min']
        
        elif norm['method']=='normal':
            return (data * stats['std']) + stats['mean']