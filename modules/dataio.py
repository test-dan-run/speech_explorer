from typing import Tuple, Set, List, DefaultDict
from pathlib import Path
from os.path import expanduser
from tqdm import tqdm
import json
import numpy as np

def convert_absolute_path(audio_filepath: str, manifest_path: str) -> str:
    '''Returns absolute path of an audio file.
    '''
    audio_filepath = Path(audio_filepath)

    if not audio_filepath.is_file() and not audio_filepath.is_absolute():
        # Checks if a file exists at the the audio_filepath,
        # otherwise, assume path is relative to directory where manifest is stored.
        manifest_dir = Path(manifest_path).parent
        audio_filepath = manifest_dir / audio_filepath

    return expanduser(audio_filepath)

def load_external_vocab(vocab_path: str) -> Set[str]:

    with open(vocab_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    words = [line.strip() for line in lines]

    return set(words)

# load data from JSON manifest file
def load_data(
    manifest_path: str, 
    vocab_path: str = None,
    ) -> Tuple[List, Set, bool]:

    ext_vocab = load_external_vocab(vocab_path) if vocab_path is not None else None
    with open(manifest_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    item_data: List = []
    for line in tqdm(lines):
        item = json.loads(line)
        if type(item['text']) is not str:
            item['text'] = ''
        item_data.append(item)
    
    metrics_available = True if 'pred_text' in item_data[0] else False   

    return item_data, ext_vocab, metrics_available
