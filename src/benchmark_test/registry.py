from benchmark_test.benchmark.mmlu import (
    load_test_data as mmlu_load_test_data,
    extract_answer as mmlu_extract_answer,
)
from benchmark_test.benchmark.gsm8k import (
    load_test_data as gsm8k_load_test_data,
    extract_answer as gsm8k_extract_answer,
)
from benchmark_test.benchmark.arc import (
    load_test_data as arc_load_test_data,
    extract_answer as arc_extract_answer,
)
from benchmark_test.benchmark.hellaswag import (
    load_test_data as hellaswag_load_test_data,
    extract_answer as hellaswag_extract_answer,
)
from benchmark_test.benchmark.truthful_qa import (
    load_test_data as truthfulqa_load_test_data,
    extract_answer as truthfulqa_extract_answer,
)
from benchmark_test.benchmark.winogrande import (
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



# English varieties
DIALECTS = [
    "aave_rerun",
    "irish_rerun",
    "australian_rerun",
    "bhamanian_rerun",
    "east_anglian_rerun",
    "appalachian_rerun",
    "southeast_england_rerun",
    "australian_vernacular_rerun",
    "north_england_rerun",
    "southwest_england_rerun",
    "newfoundland_rerun",
    "manx_rerun",
    "new_zealand_rerun",
    "ozark_rerun",
    "scottish_rerun",
    "southeast_american_rerun",
    "cunha_rerun",
    "welsh_rerun",
]
ESL = [
    "A_arabic_rerun",
    "A_french_rerun",
    "A_german_rerun",
    "A_italian_rerun",
    "A_japanese_rerun",
    "A_chinese_mandarin_rerun",
    "A_portuguese_rerun",
    "A_russian_rerun",
    "A_spanish_rerun",
    "A_turkish_rerun",
    "B_arabic_rerun",
    "B_french_rerun",
    "B_german_rerun",
    "B_italian_rerun",
    "B_japanese_rerun",
    "B_chinese_mandarin_rerun",
    "B_portuguese_rerun",
    "B_russian_rerun",
    "B_spanish_rerun",
    "B_turkish_rerun",
]
