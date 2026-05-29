# src/dataset.py

import random
from typing import Dict, List, Tuple
from src.config import COMMAND_TEMPLATES, OOS_TEMPLATES, ALL_COMMANDS, CORE_COMMANDS

# Seed for reproducibility
random.seed(42)

# Phonetic swaps for Indian Accent simulation
INDIAN_PHONETIC_REPLACEMENTS: Dict[str, List[str]] = {
    "volume": ["wolume", "wolyum", "volum"],
    "vehicle": ["vechicle", "vehical", "wehicle", "wehecle"],
    "brightness": ["brite-ness", "brightnes", "brightnesh"],
    "previous": ["preveous", "privius", "previos"],
    "music": ["musig", "museek", "mujic"],
    "decline": ["declinee", "dekline"],
    "disturb": ["desturb", "distarb"],
    "the": ["de", "ze", "d"]
}

INDIAN_PHRASING_PATTERNS: Dict[int, List[str]] = {
    5: ["music play", "song play please", "music start na"],
    6: ["music pause ya", "song stop please", "music pause na"],
    7: ["next track please", "next song play", "skip track na"],
    8: ["previous track please", "previous song play", "last song replay na"],
    9: ["volume increase please", "volume up ya", "volume increase na"],
    10: ["volume decrease please", "volume down ya", "volume decrease na"],
    11: ["brightness increase please", "brightness up ya", "brightness increase na"],
    12: ["brightness decrease please", "brightness down ya", "brightness decrease na"],
    13: ["car start please", "engine turn on na", "car start na"],
    14: ["car stop please", "engine turn off na", "car stop na"]
}

FILLERS: List[str] = ["um", "uh", "please", "can you", "hey", "like", "just", "please do one thing"]
INDIAN_POST_FILLERS: List[str] = ["na", "ya", "re", "please"]

def drop_articles(text: str) -> str:
    """Drops articles ('the', 'a', 'an') to simulate fast speech or ASR omission."""
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in {"the", "a", "an"}]
    return " ".join(filtered_words)

def add_fillers(text: str) -> str:
    """Prepends common filler words to simulate natural spoken speech."""
    filler = random.choice(FILLERS)
    if filler == "please do one thing":
        return f"{filler} {text}"
    return f"{filler} {text}"

def apply_indian_accent_phonetics(text: str) -> str:
    """Simulates Indian accent phonetic transcriptions (e.g. w/v swaps, spelling errors)."""
    words = text.split()
    augmented_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in INDIAN_PHONETIC_REPLACEMENTS:
            # 70% chance to swap with one of the Indian phonetic variants
            if random.random() < 0.7:
                augmented_words.append(random.choice(INDIAN_PHONETIC_REPLACEMENTS[w_lower]))
                continue
        # General character level swaps
        if "th" in w_lower and random.random() < 0.5:
            w_lower = w_lower.replace("th", random.choice(["d", "z"]))
        if "w" in w_lower and random.random() < 0.4:
            w_lower = w_lower.replace("w", "v")
        elif "v" in w_lower and random.random() < 0.4:
            w_lower = w_lower.replace("v", "w")
        augmented_words.append(w_lower)
    return " ".join(augmented_words)

def apply_indian_syntax_and_fillers(cmd_id: int, text: str) -> str:
    """Applies Indian English syntax ordering or appends local particles like 'na'/'ya'."""
    # 40% chance of subject-object-verb/reordered phrasing if available
    if cmd_id in INDIAN_PHRASING_PATTERNS and random.random() < 0.4:
        return random.choice(INDIAN_PHRASING_PATTERNS[cmd_id])
    
    # 50% chance of appending local particle
    if random.random() < 0.5:
        post_filler = random.choice(INDIAN_POST_FILLERS)
        return f"{text} {post_filler}"
    return text

def apply_phonetic_typos(text: str) -> str:
    """Simulates random character-level typing or ASR transcription typos."""
    if len(text) < 5:
        return text
    chars = list(text)
    idx = random.randint(0, len(chars) - 1)
    if chars[idx].isalpha():
        # 50% chance of double character, 50% chance of swap
        if random.random() < 0.5:
            chars.insert(idx, chars[idx])
        else:
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

def generate_augmented_variants(cmd_id: int, template: str) -> List[str]:
    """Generates various augmented versions of a seed template using noise simulation."""
    variants = []
    
    # 1. Dropped articles
    dropped = drop_articles(template)
    if dropped != template:
        variants.append(dropped)
        
    # 2. Filler words
    variants.append(add_fillers(template))
    if dropped != template:
        variants.append(add_fillers(dropped))
        
    # 3. Indian Accent phonetics
    variants.append(apply_indian_accent_phonetics(template))
    variants.append(apply_indian_accent_phonetics(dropped))
    
    # 4. Indian phrasing and post-fillers
    variants.append(apply_indian_syntax_and_fillers(cmd_id, template))
    variants.append(apply_indian_syntax_and_fillers(cmd_id, dropped))
    
    # 5. Phonation typos
    variants.append(apply_phonetic_typos(template))
    
    # Clean duplicates
    return list(set([v.strip().lower() for v in variants if v.strip()]))

def generate_splits(include_extension: bool = True) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Generates training data, clean test data, and noisy test data splits.
    
    Returns:
        Tuple containing:
            - train_data: List of dicts with 'text' and 'label' (command name or 'None')
            - clean_test: List of dicts for evaluation on clean inputs
            - noisy_test: List of dicts with 'text', 'label', and 'noise_type' metadata
    """
    train_data: List[Dict[str, str]] = []
    clean_test: List[Dict[str, str]] = []
    noisy_test: List[Dict[str, str]] = []
    
    commands_to_use = ALL_COMMANDS if include_extension else CORE_COMMANDS
    
    for cmd_id, cmd_name in commands_to_use.items():
        templates = COMMAND_TEMPLATES[cmd_id]
        
        # Split templates: 70% train, 30% test
        split_idx = max(1, int(len(templates) * 0.7))
        train_templates = templates[:split_idx]
        test_templates = templates[split_idx:]
        
        # 1. Populate train with clean templates and their augmentations
        for t in train_templates:
            train_data.append({"text": t, "label": cmd_name})
            # Generate augmentations for training
            for variant in generate_augmented_variants(cmd_id, t):
                train_data.append({"text": variant, "label": cmd_name})
                
        # 2. Populate clean test set
        for t in test_templates:
            clean_test.append({"text": t, "label": cmd_name})
            
        # 3. Populate noisy test set (specifically from test templates to prevent leakage)
        for t in test_templates:
            # Dropped article noise
            dropped = drop_articles(t)
            if dropped != t:
                noisy_test.append({"text": dropped, "label": cmd_name, "noise_type": "dropped_articles"})
            
            # Fillers noise
            noisy_test.append({"text": add_fillers(t), "label": cmd_name, "noise_type": "filler_words"})
            
            # Indian phonetics noise
            noisy_test.append({"text": apply_indian_accent_phonetics(t), "label": cmd_name, "noise_type": "indian_phonetics"})
            
            # Indian phrasing/syntax noise
            noisy_test.append({"text": apply_indian_syntax_and_fillers(cmd_id, t), "label": cmd_name, "noise_type": "indian_phrasing"})
            
            # Typo noise
            noisy_test.append({"text": apply_phonetic_typos(t), "label": cmd_name, "noise_type": "phonetic_typos"})
            
    # Generate Out-Of-Scope (OOS) data
    # 70% train, 30% test split
    random.shuffle(OOS_TEMPLATES)
    oos_split = int(len(OOS_TEMPLATES) * 0.7)
    
    for oos_text in OOS_TEMPLATES[:oos_split]:
        train_data.append({"text": oos_text, "label": "None"})
        
    for oos_text in OOS_TEMPLATES[oos_split:]:
        clean_test.append({"text": oos_text, "label": "None"})
        # Noisy OOS can just be general OOS variations
        noisy_test.append({"text": oos_text, "label": "None", "noise_type": "out_of_scope"})
        
    # Deduplicate splits
    train_data = [dict(t) for t in {tuple(d.items()) for d in train_data}]
    clean_test = [dict(t) for t in {tuple(d.items()) for d in clean_test}]
    
    # Shuffle for fairness
    random.shuffle(train_data)
    random.shuffle(clean_test)
    random.shuffle(noisy_test)
    
    return train_data, clean_test, noisy_test

if __name__ == "__main__":
    # Test dataset generator output
    train, test_clean, test_noisy = generate_splits()
    print(f"Generated {len(train)} training samples.")
    print(f"Generated {len(test_clean)} clean test samples.")
    print(f"Generated {len(test_noisy)} noisy test samples.")
    print("\nSample training points:")
    for i in range(5):
        print(f" - [{train[i]['label']}] -> '{train[i]['text']}'")
