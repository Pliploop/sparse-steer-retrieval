from transformers import AutoTokenizer
import torch

# Extract subjects from prompt
def extract_subjects(prompt, start_subject="<subject>", end_subject="</subject>"):
    subjects = []
    start = 0
    while True:
        start = prompt.find(start_subject, start)
        if start == -1:
            break
        end = prompt.find(end_subject, start)
        if end == -1:
            break
        subjects.append(prompt[start + len(start_subject):end])
        start = end + len(end_subject)
    return subjects

# Remove delimiters from prompt
def remove_delimiters(prompt, start_subject="<subject>", end_subject="</subject>"):
    return prompt.replace(start_subject, "").replace(end_subject, "")

# Find subject positions in clean prompt
def find_subject_positions(clean_prompt, subjects):
    positions = []
    for subject in subjects:
        start = 0
        while True:
            start = clean_prompt.find(subject, start)
            if start == -1:
                break
            end = start + len(subject)
            positions.append([start, end])
            start = end
    return positions

# Create binary mask
def create_binary_mask(clean_prompt, adjusted_positions, offsets):
    mask = [0] * len(offsets)
    for start, end in adjusted_positions:
        for i, (token_start, token_end) in enumerate(offsets):
            if token_start >= start and token_end <= end:
                mask[i] = 1
    return mask

# Process list of prompts
def create_binary_mask_from_list(dirty_prompts, offsets, start_subject="<subject>", end_subject="</subject>"):
    
    clean_prompts = [remove_delimiters(prompt, start_subject, end_subject) for prompt in dirty_prompts]
    
    
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.tolist()
    
    masks = []
    for dirty_prompt, clean_prompt, offset in zip(dirty_prompts, clean_prompts, offsets):
        subjects = extract_subjects(dirty_prompt, start_subject, end_subject)
        adjusted_positions = find_subject_positions(clean_prompt, subjects)
        mask = create_binary_mask(clean_prompt, adjusted_positions, offset)
        masks.append(mask)
    return {
        "masks": masks,
        "clean_prompts": clean_prompts
    }

