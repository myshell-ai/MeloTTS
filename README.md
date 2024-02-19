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
| English (Indian)       | [Link](https://myshell-public-repo-hosting.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
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

### Spanish
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0
text = "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante."
model = TTS(language='ES')
speaker_ids = model.hps.data.spk2id

output_path = 'es.wav'
model.tts_to_file(text, speaker_ids['ES'], output_path, speed=speed)
```

### French
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0
text = "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante."
model = TTS(language='FR')
speaker_ids = model.hps.data.spk2id

output_path = 'fr.wav'
model.tts_to_file(text, speaker_ids['FR'], output_path, speed=speed)
```

### Chinese
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0
text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH')
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)
```

### Japanese
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0
text = "彼は毎朝ジョギングをして体を健康に保っています。"
model = TTS(language='JP')
speaker_ids = model.hps.data.spk2id

output_path = 'jp.wav'
model.tts_to_file(text, speaker_ids['JP'], output_path, speed=speed)
```

### Korean
```python
from MyShellTTSBase.api import TTS

# Speed is adjustable
speed = 1.0
text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
model = TTS(language='KR')
speaker_ids = model.hps.data.spk2id

output_path = 'kr.wav'
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
```

## License
This library is under MIT License. Free for both commercial and non-commercial use.

## Acknowledgement
This implementation is based on several excellent projects, [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), and [VITS2](https://github.com/daniilrobnikov/vits2). We appreciate their awesome work!