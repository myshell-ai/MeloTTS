## Install and Use Locally

### Table of Content
- [Linux and macOS Install](#linux-and-macos-install)
- [Docker Install for Windows and macOS](#docker-install)
- [Usage](#usage)
  - [Web UI](#webui)
  - [CLI](#cli)
  - [Python API](#python-api)

### Linux and macOS Install
The repo is developed and tested on `Ubuntu 20.04` and `Python 3.9`.
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```
If you encountered issues in macOS install, try the [Docker Install](#docker-install)

### Docker Install
To avoid compatibility issues, for Windows users and some macOS users, we suggest to run via Docker. Ensure that [you have Docker installed](https://docs.docker.com/engine/install/).

**Build Docker**

This could take a few minutes.
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
docker build -t melotts . 
```

**Run Docker**
```bash
docker run -it -p 8888:8888 melotts
```
If your local machine has GPU, then you can choose to run:
```bash
docker run --gpus all -it -p 8888:8888 melotts
```
Then open [http://localhost:8888](http://localhost:8888) in your browser to use the app.

## Usage

### WebUI

The WebUI supports muliple languages and voices. First, follow the installation steps. Then, simply run:

```bash
melo-ui
# Or: python melo/app.py
```

### CLI

You may use the MeloTTS CLI to interact with MeloTTS. The CLI may be invoked using either `melotts` or `melo`. Here are some examples:

**Read English text:**

```bash
melo "Text to read" output.wav
```

**Specify a language:**

```bash
melo "Text to read" output.wav --language EN
```

**Specify a speaker:**

```bash
melo "Text to read" output.wav --language EN --speaker EN-US
melo "Text to read" output.wav --language EN --speaker EN-AU
```

The available speakers are: `EN-Default`, `EN-US`, `EN-BR`, `EN_INDIA` `EN-AU`.

**Specify a speed:**

```bash
melo "Text to read" output.wav --language EN --speaker EN-US --speed 1.5
melo "Text to read" output.wav --speed 1.5
```

**Use a different language:**

```bash
melo "text-to-speech 领域近年来发展迅速" zh.wav -l ZH
```

**Load from a file:**

```bash
melo file.txt out.wav --file
```

The full API documentation may be found using:

```bash
melo --help
```

### Python API

#### English with Multiple Accents

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

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

# Default accent
output_path = 'en-default.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)

```

#### Spanish
```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can also change to cuda:0
device = 'cpu'

text = "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante."
model = TTS(language='ES', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'es.wav'
model.tts_to_file(text, speaker_ids['ES'], output_path, speed=speed)
```

#### French

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante."
model = TTS(language='FR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'fr.wav'
model.tts_to_file(text, speaker_ids['FR'], output_path, speed=speed)
```

#### Chinese

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)
```

#### Japanese

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "彼は毎朝ジョギングをして体を健康に保っています。"
model = TTS(language='JP', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'jp.wav'
model.tts_to_file(text, speaker_ids['JP'], output_path, speed=speed)
```

#### Korean

```python
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'kr.wav'
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)
```
