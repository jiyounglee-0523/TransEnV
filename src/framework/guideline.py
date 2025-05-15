import os
import pandas as pd

from registry.guidline import *
from registry.framework import CEFR_GRAMMAR_FEATURE_PATH
from utils import colorstr
from utils.filesys_utils import json_load



def dialect_feature(dialect, data_path):
    ewave = pd.read_csv(os.path.join(data_path, 'ewave/ewave.csv'))
    linguistic_features = ewave[(ewave['Language_ID'] == dialect) & (ewave['Value'] == 'A')]['Parameter_ID'].tolist()
    return linguistic_features



def cefr_feature(cefr_level):
    online_profile = pd.read_excel(CEFR_GRAMMAR_FEATURE_PATH)

    mapping = {
        'A1': 'A', 'A2': 'A',
        'B1': 'B', 'B2': 'B',
        'C1': 'C', 'C2': 'C'
    }
    
    online_profile['Level'] = online_profile['Level'].replace(mapping)

    levels = ['A', 'B', 'C']
    input_index = levels.index(cefr_level)
    filtered_levels = levels[input_index + 1:]
    
    linguistic_features = online_profile[online_profile['Level'].isin(filtered_levels)]['Can-do statement'].tolist()
    
    return linguistic_features



def return_guideline(task_config, dataset_name, data_path):
    if (task_config.task_name == 'english_dialect') & (task_config.dialect is not None):
        file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example.json')
        guideline = json_load(file_path)
        linguistic_features = dialect_feature(dialect=task_config.dialect.strip("'\""), data_path=data_path)

        if dataset_name in ['mmlu', 'hellaswag']:
            linguistic_features = [l for l in linguistic_features if l in DIALECT_FEATURE_LIST]

        guideline = [(g['feature'][3:-3], g['guideline']) for g in guideline if g['feature'][3:-3] in linguistic_features]

    elif (task_config.task_name == 'L1') & (task_config.l1 is not None):
        l1_file_path = os.path.join(data_path, 'assets/guidelines/python_grammar_error.json')
        guideline = json_load(l1_file_path)
        l1_linguistic_features = L1_GRAMMARERROR[task_config.l1]
        guideline = [(g['grammar_error'], g['guideline']) for g in guideline if g['grammar_error'] in l1_linguistic_features]


    elif (task_config.task_name == 'cefr') & (task_config.cefr_level is not None):
        cefr_file_path = os.path.join(data_path, 'assets/guidelines/orig_generated_guideline_wo_example_grammar_error.json')
        guideline = json_load(cefr_file_path)
        cefr_linguistic_features = CEFR_ERROR[task_config.cefr_level]
        guideline = [(g['feature'][1:-1].strip(), g['guideline']) for g in guideline if g['feature'][1:-1].strip() in cefr_linguistic_features]
        
    else:
        raise NotImplementedError(f'Please double check the task name, we got {task_config.task_name}')

    assert len(guideline) != 0, colorstr("red", "Guideline Empty!")

    return guideline
