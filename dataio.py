from typing import Tuple, Set
from pathlib import Path
from datetime import datetime
import os
from os.path import expanduser

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
    disable_caching: bool = False, 
    estimate_audio: bool = False, 
    vocab_path: str = None,
    ) -> Tuple:

    if vocab_path is not None:
        vocabulary_ext = load_external_vocab(vocab_path)

    if not disable_caching:
        pickle_filename = manifest_path.split('.json')[0]
        json_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(manifest_path))
        timestamp = json_mtime.strftime('%Y%m%d_%H%M')
        pickle_filename += '_' + timestamp + '.pkl'
        if os.path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as f:
                data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available = pickle.load(f)
            if vocab is not None:
                for item in vocabulary_data:
                    item['OOV'] = item['word'] not in vocabulary_ext
            if estimate_audio:
                for item in data:
                    filepath = absolute_audio_filepath(item['audio_filepath'], manifest_path)
                    signal, sr = librosa.load(path=filepath, sr=None)
                    bw = eval_bandwidth(signal, sr)
                    item['freq_bandwidth'] = int(bw)
                    item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
            with open(pickle_filename, 'wb') as f:
                pickle.dump(
                    [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                    f,
                    pickle.HIGHEST_PROTOCOL,
                )
            return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available

    data = []
    wer_dist = 0.0
    wer_count = 0
    cer_dist = 0.0
    cer_count = 0
    wmr_count = 0
    wer = 0
    cer = 0
    wmr = 0
    mwa = 0
    num_hours = 0
    vocabulary = defaultdict(lambda: 0)
    alphabet = set()
    match_vocab = defaultdict(lambda: 0)

    sm = difflib.SequenceMatcher()
    metrics_available = False
    with open(manifest_path, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            item = json.loads(line)
            if not isinstance(item['text'], str):
                item['text'] = ''
            num_chars = len(item['text'])
            orig = item['text'].split()
            num_words = len(orig)
            for word in orig:
                vocabulary[word] += 1
            for char in item['text']:
                alphabet.add(char)
            num_hours += item['duration']

            if 'pred_text' in item:
                metrics_available = True
                pred = item['pred_text'].split()
                measures = jiwer.compute_measures(item['text'], item['pred_text'])
                word_dist = measures['substitutions'] + measures['insertions'] + measures['deletions']
                char_dist = editdistance.eval(item['text'], item['pred_text'])
                wer_dist += word_dist
                cer_dist += char_dist
                wer_count += num_words
                cer_count += num_chars

                sm.set_seqs(orig, pred)
                for m in sm.get_matching_blocks():
                    for word_idx in range(m[0], m[0] + m[2]):
                        match_vocab[orig[word_idx]] += 1
                wmr_count += measures['hits']

            data.append(
                {
                    'audio_filepath': item['audio_filepath'],
                    'duration': round(item['duration'], 2),
                    'num_words': num_words,
                    'num_chars': num_chars,
                    'word_rate': round(num_words / item['duration'], 2),
                    'char_rate': round(num_chars / item['duration'], 2),
                    'text': item['text'],
                }
            )
            if metrics_available:
                data[-1]['pred_text'] = item['pred_text']
                if num_words == 0:
                    num_words = 1e-9
                if num_chars == 0:
                    num_chars = 1e-9
                data[-1]['WER'] = round(word_dist / num_words * 100.0, 2)
                data[-1]['CER'] = round(char_dist / num_chars * 100.0, 2)
                data[-1]['WMR'] = round(measures['hits'] / num_words * 100.0, 2)
                data[-1]['I'] = measures['insertions']
                data[-1]['D'] = measures['deletions']
                data[-1]['D-I'] = measures['deletions'] - measures['insertions']

            if estimate_audio:
                filepath = absolute_audio_filepath(item['audio_filepath'], manifest_path)
                signal, sr = librosa.load(path=filepath, sr=None)
                bw = eval_bandwidth(signal, sr)
                item['freq_bandwidth'] = int(bw)
                item['level_db'] = 20 * np.log10(np.max(np.abs(signal)))
            for k in item:
                if k not in data[-1]:
                    data[-1][k] = item[k]

    vocabulary_data = [{'word': word, 'count': vocabulary[word]} for word in vocabulary]
    if vocab is not None:
        for item in vocabulary_data:
            item['OOV'] = item['word'] not in vocabulary_ext

    if metrics_available:
        wer = wer_dist / wer_count * 100.0
        cer = cer_dist / cer_count * 100.0
        wmr = wmr_count / wer_count * 100.0

        acc_sum = 0
        for item in vocabulary_data:
            w = item['word']
            word_accuracy = match_vocab[w] / vocabulary[w] * 100.0
            acc_sum += word_accuracy
            item['accuracy'] = round(word_accuracy, 1)
        mwa = acc_sum / len(vocabulary_data)

    num_hours /= 3600.0

    if not disable_caching:
        with open(pickle_filename, 'wb') as f:
            pickle.dump(
                [data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available],
                f,
                pickle.HIGHEST_PROTOCOL,
            )

    return data, wer, cer, wmr, mwa, num_hours, vocabulary_data, alphabet, metrics_available
