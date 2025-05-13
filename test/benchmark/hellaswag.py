from datasets import load_dataset
from dotenv import load_dotenv
import re
import os
import ast

load_dotenv()

letter2index = dict(A=0, B=1, C=2, D=3, E=4)

expected_size_dict = {
    'dialect': 10042,
    'A': 7593,
    'B': 9903,
}

# {'ind': 117,
#  'activity_label': 'High jump',
#  'ctx_a': 'The boy lands on his back on to a red mat. The boy gets up from the mat.',
#  'ctx_b': 'the boy',
#  'ctx': 'The boy lands on his back on to a red mat. The boy gets up from the mat. the boy',
#  'endings': ['starts doing spins.',
#   'celebrates by clapping and flexing both arms.',
#   'is dancing on the mat.',
#   'does jump jacks on his stick.'],
#  'source_id': 'activitynet~v_-zHX3Gdx6I4',
#  'split': 'val',
#  'split_type': 'zeroshot',
#  'label': '1'}


def gen_prompt(instance, idx, is_cot=False):
    prompt = "As an expert in commonsense reasoning, finish the following question.\n\n"
    prompt += f"Category: {instance['activity_label']}"
    prompt += f"\nQuestion: {instance['ctx']}"
    letters = list(letter2index.keys())
    for i, choice in enumerate(instance["endings"]):
        prompt += f"\n{letters[i]}. {choice}"
    prompt += "\nAnswer:"
    if is_cot:
        prompt += f" Let's think step by step."
    instance["answer"] = instance["label"]
    instance["prompt"] = prompt
    instance["original_idx"] = idx
    return instance


def preprocess(dataset, args):
    return dataset.map(lambda x, idx: gen_prompt(x, idx, args.cot), with_indices=True, load_from_cache_file=False, keep_in_memory=True)


def load_hellaswag(args):
    dataset = load_dataset(
        "Rowan/hellaswag", 
        "default", 
        split="validation",
        cache_dir=os.environ.get("DATA_DIR"),
    )
    dataset = preprocess(dataset, args)
    return dataset


def load_test_data(args):
    if args.data_path == "hellaswag":
        return load_hellaswag(args)

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
        if example["ctx"] is not None:  # type: ignore
            data_size += 1

    if data_size != expected_data_size:
        print(f"Expected {expected_data_size} HellaSwag examples but got {data_size} examples.")
        exit()

    def str2list(example):
        example["endings"] = ast.literal_eval(example["endings"])
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
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        letters = find_letters(answer)
        if letters:
            return letter2index.get(letters[0], -1)
    else:
        letters = find_letters(x)
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
