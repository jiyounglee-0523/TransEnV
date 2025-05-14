from datasets import load_dataset
from dotenv import load_dotenv
import re
import os
from torch.utils.data import Dataset, DataLoader, Subset


load_dotenv(override=True)

expected_size_dict = {
    'dialect': 1319,
    'A': 1219,
    'B': 1315,
}


def gen_prompt(instance, idx, is_cot=False):
    prompt = (
        "As an expert problem solver solve the following mathematical questions.\n\n"
    )
    prompt += f"Question: {instance['question']}"
    prompt += "\nAnswer:"
    if is_cot:
        prompt += f" Let's think step by step."
    if type(instance["answer"]) == str:
        instance["answer"] = instance["answer"].split("#### ")[1].replace(",", "")
    instance["prompt"] = prompt
    instance["original_idx"] = idx
    return instance


def preprocess(dataset, args):
    return dataset.map(lambda x, idx: gen_prompt(x, idx, args.cot), with_indices=True, load_from_cache_file=False, keep_in_memory=True)
    test = []
    for instance in dataset:
        instance["answer"] = instance["answer"].split("#### ")[1].replace(",", "")
        instance["prompt"] = gen_prompt(instance, args.cot)
        test.append(instance)
    return test


def load_gsm8k(args):
    dataset = load_dataset(
        "openai/gsm8k", 
        "main",
        split="test",
        cache_dir=os.environ.get("DATA_DIR"),
    )
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "gsm8k":
        return load_gsm8k(args)

    dataset = load_dataset(
        "csv",
        data_files={"test": args.data_path},
        split="test",
        cache_dir=args.cache_dir,
        keep_in_memory=True,
    )

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
        print(f"Expected {expected_data_size} examples but got {data_size} examples.")
        exit()

    return preprocess(dataset, args)


def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    numbers = re.compile(
        r"-?[\d,]*\.?\d+",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str, answer_delimiter: str = "nswer is") -> str:
    if answer_delimiter == "":
        letters = find_numbers(x)
        if letters:
            return letters[0]

    elif answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]
        else:
            answer = x.split(answer_delimiter)[-2]
            numbers = find_numbers(answer)
            if numbers:
                return numbers[0]
        
    return ""


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(",", "")


def extract_answer(outputs):
    raw_outputs = []
    extracted_outputs = []
    if isinstance(outputs, str):
        if 'boxed' in outputs:
            short = maybe_remove_comma(find_number(outputs, 'boxed'))
            return short
        else:
            short = maybe_remove_comma(find_number(outputs))
            return short

    elif hasattr(outputs, "text"):
        text = outputs.text
        short = maybe_remove_comma(find_number(text))
        raw_outputs.append(text)
        extracted_outputs.append(short)
        return raw_outputs, extracted_outputs

    else:
        for output in outputs.choices:
            try:
                text = output.text
            except AttributeError:
                text = output.message.content
            short = maybe_remove_comma(find_number(text))
            raw_outputs.append(text)
            extracted_outputs.append(short)
        return raw_outputs, extracted_outputs
