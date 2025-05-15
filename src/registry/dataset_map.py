import os
from datasets import load_dataset

from benchmark.mmlu import (
    mmlu_dataloader,
    load_mmlu_test_dataset,
    load_test_data as mmlu_load_test_data,
    extract_answer as mmlu_extract_answer,
)
from benchmark.gsm8k import (
    gsm8k_dataloader,
    load_test_data as gsm8k_load_test_data,
    extract_answer as gsm8k_extract_answer,
)
from benchmark.arc import (
    arc_dataloader,
    load_test_data as arc_load_test_data,
    extract_answer as arc_extract_answer,
)
from benchmark.hellaswag import (
    hellaswag_dataloader,
    load_test_data as hellaswag_load_test_data,
    extract_answer as hellaswag_extract_answer,
)
from benchmark.truthful_qa import (
    truthfulqa_dataloader,
    load_test_data as truthfulqa_load_test_data,
    extract_answer as truthfulqa_extract_answer,
)
from benchmark.winogrande import (
    winogrande_dataloader,
    load_test_data as winogrande_load_test_data,
    extract_answer as winogrande_extract_answer,
)



# Function definition of each benchmark
MAIN_FUNCS = {
    "mmlu": (mmlu_load_test_data, mmlu_extract_answer, 13436, "question"),
    "gsm8k": (gsm8k_load_test_data, gsm8k_extract_answer, 1319, "question"),
    "arc": (arc_load_test_data, arc_extract_answer, 1172, "question"),
    "hellaswag": (hellaswag_load_test_data, hellaswag_extract_answer, 10042, "ctx"),
    "truthful_qa": (
        truthfulqa_load_test_data,
        truthfulqa_extract_answer,
        817,
        "question",
    ),
    "winogrande": (
        winogrande_load_test_data,
        winogrande_extract_answer,
        1267,
        "sentence",
    ),
}



# Test datasets
LOAD_TEST_DATASET = {
    'mmlu': load_mmlu_test_dataset(),
    'gsm8k': load_dataset('openai/gsm8k', 'main', split='test', cache_dir=os.environ.get("DATA_DIR", None)),
    'arc': load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', cache_dir=os.environ.get("DATA_DIR", None)),
    'hellaswag': load_dataset('Rowan/hellaswag', split='validation', cache_dir=os.environ.get("DATA_DIR", None)),
    'truthfulqa': load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation', cache_dir=os.environ.get("DATA_DIR", None)),
    'winogrande': load_dataset('allenai/winogrande', split='validation', trust_remote_code=True, name='winogrande_m', cache_dir=os.environ.get("DATA_DIR", None)),
}



# Dataloaders
DATASET_MAPPING = {
    'mmlu': mmlu_dataloader,
    'gsm8k': gsm8k_dataloader,
    'arc': arc_dataloader,
    'hellaswag': hellaswag_dataloader,
    'truthfulqa': truthfulqa_dataloader,
    'winogrande': winogrande_dataloader,
}
