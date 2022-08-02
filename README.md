# Speech Explorer

This is just a minimal repository for NVIDIA NeMo's [Speech Explorer](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/speech_data_explorer.html). 

## Set Up Environment
```bash
python3 -m venv ~/venv/explorer
source ~/venv/explorer/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Dataset Manifest
Ensure your dataset manifest is in NeMo format.
```
{"audio_filepath": "rel/path/from/manifest/audio1.wav", "text": "a b c", "duration": 3.21}
{"audio_filepath": "rel/path/from/manifest/audio2.wav", "text": "a b d", "duration": 2.34}
```

If you have predictions and wish to calculate WER, ensure your manifest has the `pred_text` attribute.

```
{"audio_filepath": "rel/path/from/manifest/audio1.wav", "text": "a b c", "duration": 3.21, "pred_text": "a b c"}
{"audio_filepath": "rel/path/from/manifest/audio2.wav", "text": "a b d", "duration": 2.34, "pred_text": "a c d"}
```

## Start exploring
```bash
python3 run.py /path/to/manifest.json
```

## Citation
```BibTeX
@article{kuchaiev2019nemo,
  title   = {Nemo: a toolkit for building ai applications using neural modules},
  author  = {Kuchaiev, Oleksii and Li, Jason and Nguyen, Huyen and Hrinchuk, Oleksii and Leary, Ryan and Ginsburg, Boris and Kriman, Samuel and Beliaev, Stanislav and Lavrukhin, Vitaly and Cook, Jack and others},
  journal = {arXiv preprint arXiv:1909.09577},
  year    = {2019}
}
```
