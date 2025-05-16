import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import argparse
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoTokenizer
from google.api_core.exceptions import ResourceExhausted, InternalServerError, DeadlineExceeded

from torch.utils.data import DataLoader, Subset

from registry.benchmark import *
from registry.dataset_map import MAIN_FUNCS
from utils import log, colorstr
from utils.data_utils import *
from utils.common import exponential_backoff



# Common 
def common_call(
        args,
        output_save_path,
        desc,
        extract_answer,
        results_dataset,
        max_tokens=2048,
    ):

    client = OpenAI(base_url=f"http://localhost:{args.api_port}/v1", api_key="EMPTY")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=os.environ.get("MODEL_DIR", None))

    noanswer_indices = []

    for idx, instance in enumerate(results_dataset):
        if instance["long_answer"] is None:
            noanswer_indices.append(idx)

    log(f"Test instances left: {len(noanswer_indices)}")

    test_dataset = Subset(TestDataset(results_dataset), noanswer_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    for batch in tqdm(test_loader, desc=desc):
        inputs = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        "content": "Do not reason for too long. If the question is a multiple choice question, answer with the option letter. If none of the given options match, you may guess or say 'none of the above.' Start your final sentence with 'The answer is '.",
                    },
                    {"role": "user", "content": p},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in batch["prompt"]
        ]

        outputs = client.completions.create(
            model=args.model,
            prompt=inputs,
            max_tokens=max_tokens,
        )

        long_answers, short_answers = extract_answer(outputs)

        if args.show_sample:
            tqdm.write("!!!!!!PROMPT!!!!!!")
            tqdm.write(inputs[0])
            tqdm.write("!!!!!!!!!!!!!!!!!!")
            tqdm.write("######OUTPUT######")
            tqdm.write(long_answers[0])
            tqdm.write("##################")
            tqdm.write("MODEL: " + str(short_answers[0]))
            tqdm.write("CORRECT: " + str(int(batch["answer"][0])))

        results_dataset = results_dataset.map(
            lambda x, idx: update_map(
                x, idx, batch["original_idx"], long_answers, short_answers
            ),
            with_indices=True,
        )

        save_dataset = results_dataset.remove_columns(["prompt", "original_idx"])
        
        save_dataset.to_csv(output_save_path)



# OpenAI API call
def openai_call(
        args,
        output_save_path,
        desc,
        extract_answer,
        results_dataset,
        max_tokens=2048,
    ):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None), base_url="https://litellm.rum.uilab.kr:8080")

    noanswer_indices = []

    for idx, instance in enumerate(results_dataset):
        if instance["long_answer"] is None:
            noanswer_indices.append(idx)

    log(f"Test instances left: {len(noanswer_indices)}")

    test_dataset = Subset(TestDataset(results_dataset), noanswer_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    for batch in tqdm(test_loader, desc=desc):
        x = batch["prompt"][0]
        messages = [
            {
                "role": "system",
                "content": "Do not reason for too long. If the question is a multiple choice question, answer with the option letter. If none of the given options match, you may guess or say 'none of the above.' Start your final sentence with 'The answer is '.",
            },
            {"role": "user", "content": x},
        ]

        try:
            completions = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
            )

            long_answers, short_answers = extract_answer(completions)

        except Exception as e:
            log(str(e), level="error")
            # break
            long_answers = ["_"]
            short_answers = [-1]

        results_dataset = results_dataset.map(
            lambda x, idx: update_map(
                x, idx, batch["original_idx"], long_answers, short_answers
            ),
            with_indices=True,
        )

        save_dataset = results_dataset.remove_columns(["prompt", "original_idx"])
        
        save_dataset.to_csv(output_save_path)



# Google GCP API call
def gemini_call(
        args,
        output_save_path,
        desc,
        extract_answer,
        results_dataset,
        max_tokens=2048,
        max_retries=5,
    ):
    
    model = genai.GenerativeModel(
            args.model,
            system_instruction="Do not reason for too long. If the question is a multiple choice question, answer with the option letter. If none of the given options match, you may guess or say 'none of the above.' Start your final sentence with 'The answer is '.",
        )
    
    noanswer_indices = []

    for idx, instance in enumerate(results_dataset):
        if instance["long_answer"] is None:
            noanswer_indices.append(idx)

    log(f"Test instances left: {len(noanswer_indices)}")

    test_dataset = Subset(TestDataset(results_dataset), noanswer_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    for batch in tqdm(test_loader, desc=desc):
        x = batch["prompt"][0]
        
        retry_count = 0
        while True:
            try:
                response = model.generate_content(x)
                long_answers, short_answers = extract_answer(response)
                break
            except (ResourceExhausted, InternalServerError, DeadlineExceeded, ValueError) as e:
                if retry_count >= max_retries:
                    log(f"\nMax retries reached. Last error: {e}", level='error')
                    raise e
                
                wait_time = exponential_backoff(retry_count)
                log(f"\n[{retry_count + 1}/{max_retries}] {type(e).__name__}: {e}. Retrying in {wait_time:.1f} seconds...", level='warning')
                time.sleep(wait_time)
                retry_count += 1
                continue

        results_dataset = results_dataset.map(
            lambda x, idx: update_map(
                x, idx, batch["original_idx"], long_answers, short_answers
            ),
            with_indices=True,
        )

        save_dataset = results_dataset.remove_columns(["prompt", "original_idx"])
        
        save_dataset.to_csv(output_save_path)



def env_setup(args):
    # Load .env variables
    load_dotenv(override=True)
    
    # GPU initialization
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    args.gpus = len(gpus.split(",")) if gpus else 1     # In-place logic
    
    # GCP Gemini API key initialization
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)



def build_experiment(args):
    # Define functions
    for data in MAIN_FUNCS:
        if data in args.data_path:
            original_data_name = data
            load_test_data, extract_answer, data_size, question_key = MAIN_FUNCS[data]
            break
    else:
        raise NotImplementedError(colorstr("red", "Please double check your `--data-path` argument.."))

    # Make directory and set a file name
    output_dir = os.path.join(args.output_dir, original_data_name)
    os.makedirs(output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    data_name = os.path.splitext(args.data_path.split("/")[-1])[0]
    filename = [model_name]
    cot_yesno = "no"

    if data_name in DIALECTS:
        filename.append("dialect")
    elif data_name in ESL:
        filename.append("l1")
    else:
        filename.append("original")
    
    filename.append(data_name)
    
    if args.cot:
        cot_yesno = "yes"
        filename.append("cot")

    output_filename = "__".join(filename) + ".csv"
    output_save_path = os.path.join(output_dir, output_filename)
    
    desc = f"[{model_name}|{original_data_name}_{data_name}|{cot_yesno} CoT]"

    if os.path.exists(output_save_path):
        args.data_path = output_save_path
        results_dataset = load_test_data(args)
    else:
        results_dataset = load_test_data(args)
        results_dataset = results_dataset.map(map_fn)
    
    return {
        "args": args,
        "output_save_path": output_save_path,
        "desc": desc,
        "extract_answer": extract_answer,
        "results_dataset": results_dataset
    }
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache directory.")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--data-path", type=str, default="mmlu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-api-retries", type=int, default=5)
    parser.add_argument("--cot", action="store_true", help="cot")
    parser.add_argument("--show-sample", "-s", action="store_true")
    args = parser.parse_args()

    # Environment setup
    env_setup(args)

    # Build experiment setup
    exp_config = build_experiment(args)

    # Execution
    if args.model in ['o1-mini-2024-09-12', 'azure/o1-mini-2024-09-12', 'azure/gpt-4o-mini-2024-07-18', 'azure/o3-mini-2025-01-31', 'o3-mini-2025-01-31']:
        openai_call(
            max_tokens=args.max_tokens,
            **exp_config
        )
    elif args.model in ['models/gemini-2.5-flash-preview-04-17', 'models/gemini-2.0-flash', 'models/gemini-1.5-flash', 'models/gemini-2.5-pro-exp-03-25', 'models/gemini-2.5-pro-preview-03-25']:
        gemini_call(
            max_tokens=args.max_tokens,
            max_retries=args.max_api_retries,
            **exp_config
        )
    else:
        common_call(
            max_tokens=args.max_tokens,
            **exp_config
        )
