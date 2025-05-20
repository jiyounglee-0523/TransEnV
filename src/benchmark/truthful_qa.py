import re
import os
import ast
from dotenv import load_dotenv
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset

from utils import log



load_dotenv(override=True)

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

letter2index = {letter: i for i, letter in enumerate(letters)}

expected_size_dict = {
    'dialect': 817,
    'A': 623,
    'B': 781,
}


def gen_prompt(instance, idx, is_cot=False):
    question = instance["question"]
    if type(instance["mc1_targets"]) == list:
        instance["mc1_targets"] = instance["mc1_targets"][0]
    key = instance["mc1_targets"]["labels"]
    prompt = "Answer the following multiple choice question.\n\n"
    prompt += f"Question: {question}"
    for i, choice in enumerate(instance["mc1_targets"]["choices"]):
        prompt += f"\n{letters[i]}. {choice}"

    prompt += "\nAnswer:"

    if is_cot:
        prompt += f"\nLet's think step by step."

    instance["answer"] = key.index(1)
    instance["prompt"] = prompt
    instance["original_idx"] = idx
    return instance


def preprocess(dataset, args):
    return dataset.map(lambda x, idx: gen_prompt(x, idx, args.cot), with_indices=True, load_from_cache_file=False, keep_in_memory=True)


def load_truthful_qa(args):
    dataset = load_dataset(
        "truthfulqa/truthful_qa",
        "multiple_choice",
        split="validation",
        cache_dir=os.environ.get("DATA_DIR", None),
    )
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "truthful_qa":
        return load_truthful_qa(args)

    dataset = load_dataset(
        "csv",
        data_files={"test": args.data_path},
        split="test",
        cache_dir=args.cache_dir,
        keep_in_memory=True,
    )

    # expected sample size
    data_type = None
    if ('dialect' in args.data_path) or ('__dialect__' in args.data_path) or ('__original__' in args.data_path):
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
        log(f"Expected {expected_data_size} TruthfulQA examples but got {data_size} examples.", level="error")
        exit()

    def str2list(example):
        example["mc1_targets"] = ast.literal_eval(example["mc1_targets"])
        example["mc2_targets"] = ast.literal_eval(example["mc2_targets"])
        return example

    return preprocess(dataset.map(str2list, load_from_cache_file=False, keep_in_memory=True), args)


def find_letters(x: str) -> list[str]:
    """Finds A, B, C, D in a string."""
    letters = re.compile(
        r"\b[A-E]\b",
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



# TruthfulQA dataloader
def truthfulqa_dataloader(batch_size, rerun_index=None, start_idx=None):

    class TruthfulQADataset(Dataset):
        def __init__(self, pad_value='None'):
            self.dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')

            if start_idx is not None:
                self.dataset = self.dataset.skip(start_idx)

            if rerun_index is not None:
                self.dataset = self.dataset.select(rerun_index)

            self.mc1_max_labels = 13
            self.mc2_max_labels = 20
            self.pad_value = pad_value

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

            mc1_target_choice = sample['mc1_targets']['choices']
            mc1_target_labels = sample['mc1_targets']['labels']
            mc2_target_choice = sample['mc2_targets']['choices']
            mc2_target_labels = sample['mc2_targets']['labels']
            
            if len(sample['mc1_targets']['choices']) < self.mc1_max_labels:
                mc1_target_choice.extend([self.pad_value] * (self.mc1_max_labels - len(mc1_target_choice)))
                mc1_target_labels.extend([0] * (self.mc1_max_labels - len(mc1_target_labels)))

            if len(sample['mc2_targets']['choices']) < self.mc2_max_labels:
                mc2_target_choice.extend([self.pad_value] * (self.mc2_max_labels - len(mc2_target_choice)))
                mc2_target_labels.extend([0] * (self.mc2_max_labels - len(mc2_target_labels)))
    
            
            return {
                'question': sample['question'],
                'mc1_targets': {'choices': mc1_target_choice, 'labels': mc1_target_labels},
                'mc2_targets': {'choices': mc2_target_choice, 'labels': mc2_target_labels}
            }

    test_dataset = TruthfulQADataset()
    
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return test_loader
