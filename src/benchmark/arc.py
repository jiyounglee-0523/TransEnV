import re
import os
import ast
from dotenv import load_dotenv
from datasets import load_dataset

from torch.utils.data import DataLoader, Dataset

from utils import log



load_dotenv(override=True)

letter2index = dict(A=0, B=1, C=2, D=3, E=4)

expected_size_dict = {
    'dialect': 1172,
    'A': 774,
    'B': 1132,
}

def gen_prompt(instance, idx, is_cot=False):
    key = instance["answerKey"]
    if key in "ABCDE":
        instance["answer"] = letter2index[key]
    elif key in "12345":
        instance["answer"] = int(key) - 1
    else:
        raise ValueError(f"Invalid answer key: {key}")
    if 'text' in instance["choices"]:
        instance["choices"] = instance["choices"]["text"]
    prompt = f"As an expert problem solver solve the following science questions.\n\n"
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


def load_arc(args):
    dataset = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split="test",
        cache_dir=os.environ.get("DATA_DIR", None),
    )
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "arc":
        return load_arc(args)

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
        log(f"Expected {expected_data_size} ARC examples but got {data_size} examples.", level="error")
        exit()

    def str2list(example):
        example["choices"] = ast.literal_eval(example["choices"])
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



# AI2_ARC dataloader
def arc_dataloader(batch_size, rerun_index=None, start_idx=None):

    class ARCDataset(Dataset):
        def __init__(self, max_labels=5, pad_value='None'):
            self.dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')

            if start_idx is not None:
                self.dataset = self.dataset.skip(start_idx)

            if rerun_index is not None:
                self.dataset = self.dataset.select(rerun_index)

            self.max_labels = max_labels
            self.pad_value = pad_value

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            choice_text = sample['choices']['text']
            choice_label = sample['choices']['label']
            
            if len(sample['choices']['text']) < self.max_labels:
                choice_text.extend([self.pad_value] * (self.max_labels - len(choice_text)))
                choice_label.extend([self.pad_value] * (self.max_labels - len(choice_label)))
            
            return {
                'id': sample['id'],
                'question': sample['question'],
                'choices': {'text': choice_text, 'label': choice_label},
                'answerKey': sample['answerKey']
            }

    test_dataset = ARCDataset()
    
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return test_loader
