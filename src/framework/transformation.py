import copy
import random

from registry.prompt import *
from utils.guidline_utils import *



def framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "user", "content": system_message},
        {"role": "assistant", "content": "Well Understood."},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
        ]

    return message



def transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config):
    """
    sentence (list of string) where list size is equal to batch size
    """

    if type(sentence) is tuple:     # tuple이면 list로 바꾸기
        sentence = list(sentence)

    orig_sentence = copy.deepcopy(sentence)

    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]  # rules that are answered yes to all identification questions
    mid_transformed_sentences = [[] for _ in range(len(sentence))] # transformed sentences that are transformed by applied rules
    judge_responses = [[] for _ in range(len(sentence))] # judge response to each transformed sentence
    transformed_sentences = [[] for _ in range(len(sentence))] # final transformed sentence

    # shuffle guideline
    random.shuffle(guideline)

    for i in range(len(guideline)):
        feature = guideline[i][0]
        input_prompt = framework_application(guideline=guideline[i], task=task_config.task_name)

        batch_input = [input_prompt + [{"role": 'user', "content": f"**Original Sentence:** {s}"}] for s in sentence]
        chat_batch_input = list()

        for input in batch_input:
            text = tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            chat_batch_input.append(text)

        responses = client.completions.create(
            model=model_config.model_name,
            prompt=chat_batch_input,
            **sampling_params
            )
        
        for num, response in enumerate(responses.choices):
            # save all responses
            whole_responses[num].append(response.text)

            if response.text is None:
                continue

            transformed_sentence = extract_transformed_sentence(response.text)
            if ('no change' in transformed_sentence.lower()) or transformed_sentence.lower() is None:
                continue

            else:
                # save the transformed sentences
                mid_transformed_sentences[num].append(transformed_sentence)

                semantic_input_prompt = semantic_check(orig_sentence[num], transformed_sentence)
                semantic_response = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=[{'role': 'user', 'content': semantic_input_prompt}]
                )

                # save judge response
                judge_responses[num].append(semantic_response.choices[0].message.content.lower())

                if 'no' in semantic_response.choices[0].message.content.lower():
                    sentence[num] = transformed_sentence
                    applied_rules[num].append(feature)
                    transformed_sentences[num].append(transformed_sentence)
        
    iter_result = list()

    for num in range(len(sentence)):
        iter_result.append({
            'orig_sentence': orig_sentence[num],
            'whole_response': whole_responses[num],
            'mid_transformed_sentences': mid_transformed_sentences[num],
            'judge_repsonse': judge_responses[num],
            'applied_rules': applied_rules[num],
            'transformed_sentences': transformed_sentences[num],
            'final_sentence': sentence[num]
        })

    return iter_result



def openai_framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
    ]

    return message



def openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config):
    sentence = sentence[0]
    orig_sentence = copy.deepcopy(sentence)

    whole_response = list()
    applied_rule = list()
    transformed_sentences = list()

    random.shuffle(guideline)

    exception_count = 0

    for i in range(len(guideline)):
        feature = guideline[i][0]
        input_prompt = openai_framework_application(guideline=guideline[i], task=task_config.task_name)
        input_prompt = input_prompt + [{"role": 'user', "content": f"**Original Sentence:** {sentence}"}]

        try:
            responses = client.chat.completions.create(
                model=model_config.model_name,
                messages=input_prompt,
                **sampling_params
            )

            response = responses.choices[0].message.content
            whole_response.append(response)

            if response is None:
                continue

            transformed_sentence = extract_transformed_sentence(response)
            

            if ('no change' in transformed_sentence.lower()) or transformed_sentence.lower() is None:
                continue

            else:
                sentence = transformed_sentence
                applied_rule.append(feature)
                transformed_sentences.append(transformed_sentence)

        except Exception as e:
            exception_count += 1
            continue

    if exception_count == len(guideline):
        sentence = orig_sentence
    
    iter_result = [{
        'orig_sentence': orig_sentence,
        'whole_response': whole_response,
        'applied_rule': applied_rule,
        'transformed_sentences': transformed_sentences,
        'final_sentence': sentence
    }]

    return iter_result
