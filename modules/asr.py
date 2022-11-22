import difflib
from collections import defaultdict
from typing import Tuple, Dict, Set, List, DefaultDict

import jiwer
import librosa
import numpy as np
import pandas as pd
import editdistance

from .common import eval_bandwidth
from .dataio import convert_absolute_path

def compute_asr_metrics(
    item_data: List,
    metrics_available: bool = False,
    estimate_audio: bool = True,
    manifest_path: str = '',
    ) -> pd.DataFrame:
    
    for item in item_data:

        if not isinstance(item['text'], str):
            item['text'] = ''
        words = item['text'].split()

        num_chars = len(item['text'])
        num_words = len(words)

        # add attributes to existing item
        item['num_words'] = num_words if num_words != 0 else 1e-9
        item['num_chars'] = num_chars if num_chars != 0 else 1e-9
        item['word_rate'] = round(num_words/item['duration'], 3)
        item['char_rate'] = round(num_chars/item['duration'], 3)

        if metrics_available:
            measures = jiwer.compute_measures(item['text'], item['pred_text'])
            item['word_dist'] = measures['substitutions'] + measures['insertions'] + measures['deletions']
            item['char_dist'] = editdistance.eval(item['text'], item['pred_text'])

            item['sub_hits'] = 0
            hypo_copy = item['text'].split()
            pred_copy = item['pred_text'].split()
            pred_copy2 = pred_copy.copy()
            for word in pred_copy2:
                if word in hypo_copy:
                    item['sub_hits'] += 1
                    hypo_copy.remove(word)
                    pred_copy.remove(word)
            for word in pred_copy:
                for word2 in hypo_copy:
                    if editdistance.eval(word, word2)/len(word) < 0.2:
                        item['sub_hits'] += 1
                        hypo_copy.remove(word2)
                        break
            
            item['hits'] = measures['hits']

            item['WER'] = round(item['word_dist']/num_words*100.0, 3)
            item['CER'] = round(item['char_dist']/num_chars*100.0, 3)
            item['WMR'] = round(measures['hits']/num_words*100.0, 3)
            item['SWMR'] = round(item['sub_hits']/num_words*100.0, 3)
            item['I'] = measures['insertions']
            item['D'] = measures['deletions']
            item['D-I'] = measures['deletions'] - measures['insertions']

        if estimate_audio:
            filepath = convert_absolute_path(item['audio_filepath'], manifest_path)
            signal, sr = librosa.load(path=filepath, sr=None)
            bandwidth = eval_bandwidth(signal, sr)
            item['freq_bandwidth'] = int(bandwidth)
            item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))

    item_df = pd.DataFrame(item_data)
    item_df.fillna('none', inplace=True)
    return item_df

def compute_global_statistics(item_df: pd.DataFrame, ext_vocab: Set, metrics_available: bool = False) -> Tuple[Dict, Set, Set]:

    vocabulary: DefaultDict = defaultdict(lambda: 0)
    match_vocab: DefaultDict = defaultdict(lambda: 0)
    alphabet: Set = set()

    wer_dist: int = 0
    cer_dist: int = 0
    wer_count: int = 0
    cer_count: int = 0
    wmr_count: int = 0
    swmr_count: int = 0
    duration: float = 0.0

    for idx, row in item_df.iterrows():
        words = row['text'].split()
        preds = row['pred_text'].split()

        sm = difflib.SequenceMatcher()
        sm.set_seqs(words, preds)
        for m in sm.get_matching_blocks():
            for word_idx in range(m[0], m[0] + m[2]):
                match_vocab[words[word_idx]] += 1

        for word in words:
            vocabulary[word] += 1
        for char in row['text']:
            alphabet.add(char)

        duration += row['duration']
        wer_dist += row['word_dist']
        cer_dist += row['char_dist']
        wer_count += row['num_words']
        cer_count += row['num_chars']
        wmr_count += row['hits']
        swmr_count += row['sub_hits']

    vocab_data = [{'word': word, 'count': vocabulary[word]} for word in vocabulary]
    if ext_vocab is not None:
        for item in vocab_data:
            item['OOV'] = item['word'] not in ext_vocab

    global_stats = {}

    if metrics_available:
        global_stats['wer'] = wer_dist / wer_count * 100.0
        global_stats['cer'] = cer_dist / cer_count * 100.0
        global_stats['wmr'] = wmr_count / wer_count * 100.0
        global_stats['swmr'] = swmr_count / wer_count * 100.0

        acc_sum = 0
        for item in vocab_data:
            w = item['word']
            word_accuracy = match_vocab[w] / vocabulary[w] * 100.0
            acc_sum += word_accuracy
            item['accuracy'] = round(word_accuracy, 1)
        global_stats['mwa'] = acc_sum / len(vocab_data)

    global_stats['num_hours'] = round(duration / 3600.0, 2)

    return global_stats, vocab_data, alphabet
