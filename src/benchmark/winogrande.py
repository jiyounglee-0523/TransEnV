import re
import os
from dotenv import load_dotenv
from datasets import load_dataset

from torch.utils.data import DataLoader



load_dotenv(override=True)

letter2index = dict(A=0, B=1, C=2, D=3, E=4)

expected_size_dict = {
    'dialect': 1267,
    'A': 945,
    'B': 1247,
}


def gen_prompt(instance, idx, is_cot=False):
    prompt = "Choose the correctv word that fits in the blank.\n\n"
    prompt += f"\n{instance['sentence']}"
    prompt += f"\nA. {instance['option1']}"
    prompt += f"\nB. {instance['option2']}"
    prompt += "\nAnswer:"
    if is_cot:
        prompt += f" Let's think step by step."
    instance["answer"] = int(instance["answer"]) - 1
    instance["prompt"] = prompt
    instance["original_idx"] = idx
    return instance


def preprocess(dataset, args):
    return dataset.map(lambda x, idx: gen_prompt(x, idx, args.cot), with_indices=True, load_from_cache_file=False, keep_in_memory=True)


def load_winogrande(args):
    dataset = load_dataset(
        "allenai/winogrande",
        "winogrande_xs",
        split="validation",
        cache_dir=os.environ.get("DATA_DIR"),
    )
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "winogrande":
        return load_winogrande(args)

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
        if example["sentence"] is not None:  # type: ignore
            data_size += 1

    if data_size != expected_data_size:
        print(f"Expected {expected_data_size} Winogrande examples but got {data_size} examples.")
        exit()

    return preprocess(dataset, args)


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
                text = output.message.content
            short = find_letter(text)
            raw_outputs.append(text)
            extracted_outputs.append(short)
        return raw_outputs, extracted_outputs



# Winogrande dataloader
def winogrande_dataloader(batch_size, rerun_index=None, start_idx=None):
    test_dataset = load_dataset('allenai/winogrande', split='validation', trust_remote_code=True, name='winogrande_m')

    if start_idx is not None:
        test_dataset = test_dataset.skip(start_idx)

    if rerun_index is not None:
        test_dataset = test_dataset.select(rerun_index)
    
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    return test_loader
