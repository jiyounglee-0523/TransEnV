from datasets import load_dataset
from dotenv import load_dotenv
import re
import os
import ast

load_dotenv()

letter2index = dict(A=0, B=1, C=2, D=3)

expected_size_dict = {
    'dialect': 13436,
    'A': 7246,
    'B': 11970,
}

mmlu_subjects_to_load = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            # "high_school_european_history": "history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            # "high_school_us_history": "history",
            # "high_school_world_history": "history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]


def gen_prompt(instance, idx, is_cot=False):
    subject = instance["subject"].replace("_", " ")
    prompt = f"The following is a multiple choice question about {subject}.\n\n"
    prompt += f"Question: {instance['question']}"
    letters = list(letter2index.keys())
    for i, choice in enumerate(instance["choices"]):
        prompt += f"\n{letters[i]}. {choice}"
    prompt += "\nAnswer:"
    if is_cot:
        prompt += f" Let's think step by step."
    instance["prompt"] = prompt
    instance["original_idx"] = idx
    return instance


def preprocess(dataset, args):
    return dataset.map(lambda x, idx: gen_prompt(x, idx, args.cot), with_indices=True, load_from_cache_file=False, keep_in_memory=True)


def load_mmlu(args):
    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test",
        cache_dir=os.environ.get("DATA_DIR"),
    )
    # remove history
    dataset = dataset.filter(lambda x: x['subject'] in mmlu_subjects_to_load)
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "mmlu":
        return load_mmlu(args)

    dataset = load_dataset(
        "csv",
        data_files={"test": args.data_path},
        split="test",
        cache_dir=args.cache_dir,
        keep_in_memory=True,
    )

    # remove history
    dataset = dataset.filter(lambda x: x['subject'] in mmlu_subjects_to_load)

    # expected sample size
    data_type = None
    if ('/dialect/' in args.data_path) or ('__dialect__' in args.data_path) or ('__original__' in args.data_path):
        data_type = 'dialect'
    elif ('/l1/' in args.data_path):
        data_type = args.data_path.split('/')[-1][0]
    elif '__l1__' in args.data_path:
        data_type = args.data_path.split('__l1__')[-1][0]
    else:
        raise ValueError(f"Invalid data path: {args.data_path}")

    data_size = 0
    expected_data_size = expected_size_dict[data_type]


    for example in dataset:
        if example["question"] is not None:  # type: ignore
            data_size += 1

    if data_size != expected_data_size:
        print(f"Expected {expected_data_size} MMLU examples but got {data_size} examples.")
        exit()

    def str2list(example):
        example["choices"] = ast.literal_eval(example["choices"])
        
        return example

    return preprocess(dataset.map(str2list, load_from_cache_file=False, keep_in_memory=True), args)


def find_letters(x: str) -> list[str]:
    """Finds A, B, C, D in a string."""
    letters = re.compile(
        r"\b[A-D]\b",
        re.MULTILINE | re.DOTALL,
    ).findall(x)
    return letters


def find_letter(x: str, answer_delimiter: str = "nswer is"):
    if answer_delimiter == "":
        letters = find_letters(x)
        if letters:
            return letter2index.get(letters[0], -1)
    elif answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        letters = find_letters(answer)
        if letters:
            return letter2index.get(letters[0], -1)

    return -1


def extract_answer(outputs):
    raw_outputs = []
    extracted_outputs = []
    
    if isinstance(outputs, str):
        short = find_letter(outputs)
        return short
    
    elif hasattr(outputs, "text"):
        text = outputs.text
        short = find_letter(text)
        raw_outputs.append(text)
        extracted_outputs.append(short)
        return raw_outputs, extracted_outputs
    
    else:
        for output in outputs.choices:
            try:
                text = output.text
            except AttributeError:
                text = output.message.content   # openai
            short = find_letter(text)
            raw_outputs.append(text)
            extracted_outputs.append(short)
        return raw_outputs, extracted_outputs
