def return_identification_system_message(guideline_instruction): 
    return f'''
    Your task is to evaluate if the below feature is applicable to the given sentece by following the guideline.

    {guideline_instruction}

    ### Steps to Follow:
    - Answer the identification questions for the linguistic feature with either "yes" or "no."
    - Answer the questions in a very strict manner leaving no room for potential.
    - If all answers to the questions are 'yes', generate the final output as "Final answer is 'applicable'", otherwise generate "Final answer is 'inapplicable.'"
    
    ### Mandatory
    - Final sentence should be either "Final answer is 'applicable'" or "Final answer is 'inapplicable'".
    '''



def return_actionable_system_message(guideline_instruction): 
    return f'''
    Your task is to rephrase the test questions by following the guideline.

    {guideline_instruction}

    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - Provide the final transformed sentence, adhering strictly to the format and structure of the given example.

    ### Mandatory
    - Retain any <blank> without modifications.
    - Preserve the structure of the original sentence as much as possible with no information loss.
    - Follow the guideline, not considering standard English grammar.
    - Final sentence should start with '**Transformed Sentence:**'.
    '''



def return_system_message(guideline_instruction): 
    return f'''
    Your task is to rephrase the test questions by following the guideline.

    {guideline_instruction}

    ### Steps to Follow:
    1. **Identification Phase**: 
    - Answer the identification questions for the linguistic feature with either "yes" or "no."
    - Answer the questions in a very strict manner leaving no room for potential especially for <blank>.
    - Proceed to the next step only if **all** answers are "yes."
    - Otherwise, stop in identification phase with generating '**Transformed Sentence:** (No change)'.
    
    2. **Actionable Changes**: 
    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - Provide the final transformed sentence, adhering strictly to the format and structure of the given example.
    
    ### Mandatory
    - Retain any <blank> without modifications.
    - Proceed to Actional Changes only if all answers to the identification questions are 'yes'.
    - Preserve the structure of the original sentence as much as possible with no information loss.
    - Follow the guideline, not considering standard English grammar.
    - Final sentence should start with '**Transformed Sentence:**' either with sentence of (No change).
    '''



def semantic_check(sentence1, sentence2):
    return f"""
    Determine whether the meaning of Sentence 1 is significantly altered or lost in Sentence 2.
    
    ### Consideration
    - All keywords from Sentence 1 should be in Sentence 2.
    - All numbers in Sentence 1 should match with Sentence 2.
    - Focus on core information only.
    - Ignore grammar; it is not a factor for consideration.
    - Missing or incorrect prepositions should not be considered.
    - Ignore repetition of phrases. Repetition is not a factor for consideration.
    - Base your decision solely on whether essential information is missing.

    Respond with either 'yes' or 'no' only.

    Sentence 1: {sentence1}
    Sentence 2: {sentence2}
    Answer:
    """




"""
### Steps to Follow:
    1. **Identification Phase**: 
    - Answer the identification questions for the linguistic feature with either "yes" or "no."
    - Answer the questions in a very strict manner leaving no room for potential.
    - Proceed to the next step only if **all** answers are "yes."
    - Otherwise, stop in identification phase with generating '**Transformed Sentence:** (No change)'.
    
    2. **Actionable Changes**: 
    - Make only the **necessary changes** to apply the linguistic feature, ensuring no loss of information.
    - Provide the final transformed sentence, adhering strictly to the format and structure of the given example.

"""
"""
    ### Consideration
    - Keywords from sentence 1 should be in sentence 2.
    - Important information in sentence 1 should all be in sentence 2.
    - Sentence 1 and sentence 2 have the same sentence structure (no different negation).
    - Overall structures of sentence 1 and sentence 2 are identical.
""" 
    