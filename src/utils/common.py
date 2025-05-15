import random
import numpy as np

from framework.data_return import *
from registry.dataset_map import LOAD_TEST_DATASET



def exponential_backoff(retry_count:int,
                        base_delay:int=5,
                        max_delay:int=65,
                        jitter:bool=True) -> float:
    """
    Exponential backoff function for API calling.

    Args:
        retry_count (int): Retry count.
        base_delay (int, optional): Base delay seconds. Defaults to 5.
        max_delay (int, optional): Maximum delay seconds. Defaults to 165.
        jitter (bool, optional): Whether apply randomness. Defaults to True.

    Returns:
        float: Final delay time.
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    if jitter:
        delay = random.uniform(delay * 0.8, delay * 1.2)
    return delay



def save_func(to_save, save_config, dataset_config, generation_config, task_config):
    save_mapping = {
        'mmlu': return_mmlu,
        'gsm8k': return_gsm8k,
        'arc': return_arc,
        'hellaswag': return_hellaswag,
        'truthfulqa': return_truthfulqa,
        'winogrande': return_winogrande,
    }
    
    rerun_index = None
    cefr_index = None

    test_dataset = LOAD_TEST_DATASET[dataset_config.dataset_name]

    if generation_config.rerun is not None:
        rerun_index = list(np.load(generation_config.rerun))
    if task_config.task_name in ['cefr', 'L1']:
        cefr_index = list(np.load(f'assets/vocab_processed/{dataset_config.dataset_name}_{(task_config.cefr_level).lower()}.npy'))
        cefr_index = [int(index) for index in cefr_index]

    return save_mapping[dataset_config.dataset_name](test_dataset, to_save, save_config, rerun_index, cefr_index)
