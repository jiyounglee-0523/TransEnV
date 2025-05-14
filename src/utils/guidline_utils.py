import re

from utils import log



def extract_transformed_sentence(text):
    """
    Extract the transformed sentence from the provided structured text.
    
    Parameters:
        text (str): The input text containing the "Transformed Sentence" section.
    
    Returns:
        str: The transformed sentence, or a message if no transformed sentence is found.
    """
    # Define a regular expression pattern to find the transformed sentence
    pattern = r"\*\*Transformed Sentence:\*\* (.*)"
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # Return the transformed sentence if found, otherwise return a message
    if match:
        return match.group(1).strip()
    else:
        return 'No change'
    
    

def extract_guideline_l1(guideline):
    guideline_instruction = guideline.split('---')[0]
    identification_instruction = guideline.split('\n\n#### **Phase 2: Actionable Changes**')[0]
    action_instruction = '#### **Phase 2: Actionable Changes**'+guideline.split('\n#### **Phase 2: Actionable Changes**')[1]

    example = ''.join(guideline.split('---')[1:])
    example = example.replace('**Input Sentence 1:**', 'Input Sentence 1:')
    example = example.replace('**Input Sentence 2:**', 'Input Sentence 2:')
    example = example.replace('**Final broken sentence:**', '**Transformed Sentence:**')

    example1_input_sentence = example.split('Input Sentence 1:')[1].split('\n')[0].strip().strip('"')
    example1_output = ''.join(example.split('Input Sentence 1:')[1].split('\n')[1:]).split('#### Input Sentence 2:')[0]
    example1_output = example1_output.replace('Phase 2: ', '')
    example1_output = example1_output.replace('Actionable Changes:', 'Actionable Changes')
    example1_identification_output = example1_output.split('**Actionable Changes:**')[0] + "\n\n Final answer is 'applicable'."
    example1_action_output = '**Actionable Changes:**' + example1_output.split('**Actionable Changes**')[1]


    example2_input_sentence = example.split('Input Sentence 2:')[1].split('\n')[0].strip().strip('"')
    example2_output = ''.join(example.split('Input Sentence 2:')[1].split('\n')[1:])

    few_shot_example = [
        {
            "input": '**Original Sentence:**' + example1_input_sentence,
            "output": example1_output
        },
        {
            "input": '**Original Sentence:**' + example2_input_sentence,
            "output": example2_output
        }
    ]

    few_shot_identification_example = [
        {
            "input": '**Original Sentence:**' + example1_input_sentence,
            "output": example1_identification_output
        }
    ]

    few_shot_action_example = [
        {
            "input": '**Original Sentence:**' + example1_input_sentence,
            "output": example1_action_output
        }
    ]

    return guideline_instruction, few_shot_example



def extract_guideline_dialect(guideline):
    identification_instruction = guideline.split('\n### Example\n')[0].split('\n\n#### Phase 2: Actionable Changes')[0]
    action_instruction = guideline.split('\n### Example\n')[0].split('\n\n#### Phase 2: Actionable Changes')[1]
    guideline_instruction = guideline.split('\n### Example\n')[0]

    example1_input_sentence = guideline.split('\n### Example\n')[1][1:].split('\n\n')[0]
    example1_identification_output = '\n\n'.join(guideline.split('\n### Example\n')[1][1:].split('\n\n')[1:-1]).split('\n\n**Actionable Changes:**')[0] + "\n\n Final answer is 'applicable'."

    example1_action_output = '**Actionable Changes:**' + '\n\n'.join(guideline.split('\n### Example\n')[1][1:].split('\n\n')[1:-1]).split('\n\n**Actionable Changes:**')[1]

    example1_output = '\n\n'.join(guideline.split('\n### Example\n')[1][1:].split('\n\n')[1:-1])
    few_shot_identification_example = [
        {
            "input": example1_input_sentence,
            "output": example1_identification_output
        }
    ]

    few_shot_action_example = [
        {
            "input": example1_input_sentence,
            "output": example1_action_output
        }
    ]

    few_shot_example = [
        {
            "input": example1_input_sentence,
            "output": example1_output,
        }
    ]

    return guideline_instruction, few_shot_example




def extract_guideline_examples(guideline, task):
    if task == 'L1':
        extract_func = extract_guideline_l1
    elif task == 'english_dialect':
        extract_func = extract_guideline_dialect
    elif task == 'cefr':
        extract_func = extract_guideline_dialect
    else:
        log(f'Please double check your taks, we got {task}', level='error')
        raise NotImplementedError

    return extract_func(guideline)
