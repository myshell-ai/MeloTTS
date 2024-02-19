<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="200"/> 
</div>

## Introduction
MyShellTTSBase is a high-quality multi-lingual text-to-speech library. Example languages include:

| Language | Example |
| --- | --- |
| English               | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| English (American)    | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (India)       | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_000.wav) |
| Japanese              | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

The Chinese speaker supports `mixed Chinese and English`.

## Install
```bash
git clone git@github.com:myshell-ai/MyShellTTSBase.git
cd MyShellTTSBase
python setup.py install
```

## Usage

### English with Multi Accents
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN')
speaker_ids = model.hps.data.spk2id

# Default accent
output_path = 'en-default.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

# British accent
output_path = 'en-br.wav'
model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)

# Indian accent
output_path = 'en-india.wav'
model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=speed)

# Australian accent
output_path = 'en-au.wav'
model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed)

```