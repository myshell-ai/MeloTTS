<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> 
</div>

## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

## ONNX Inference Acceleration

MeloTTS supports ONNX (Open Neural Network Exchange) for potentially faster inference, especially on CPU, and for easier deployment across different platforms.

### Installation for ONNX

The necessary dependencies for ONNX inference (`onnx`, `onnxruntime`) are included in the `requirements.txt` file. You can install them along with other dependencies:

```bash
pip install -r requirements.txt
```

If you have a CUDA-compatible GPU and want to leverage GPU acceleration for ONNX, you might consider installing the GPU-enabled version of `onnxruntime`. After installing the base requirements, you can do:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
Make sure your CUDA and cuDNN versions are compatible with the chosen `onnxruntime-gpu` package.

### Exporting a Model to ONNX

To use ONNX, you first need to export a pre-trained MeloTTS model to the ONNX format. This is done using the `export_onnx.py` script.

**Example Command:**

```bash
python export_onnx.py --language EN --output_path melo_en.onnx --device cpu
```

**Arguments:**
-   `--language`: (Required) The language of the model to export (e.g., `EN`, `ZH`, `ES`). This determines which default pre-trained model is loaded for export if `--ckpt_path` is not specified.
-   `--output_path`: (Optional) The path where the exported ONNX model will be saved. Defaults to `melo.onnx`.
-   `--ckpt_path`: (Optional) Path to a specific PyTorch model checkpoint (`.pth` file). If not provided, the script will attempt to download the default pre-trained model for the specified language from HuggingFace.
-   `--device`: (Optional) The device to use for loading the PyTorch model during export (e.g., `cpu`, `cuda`). Defaults to `cpu`. The exported ONNX model itself will be runnable on various devices supported by ONNX Runtime.

### Running Inference with an ONNX Model

Once you have an ONNX model file, you can use the `melo/infer.py` script with the `--use_onnx` flag to perform text-to-speech synthesis.

**Example Command:**

```bash
python melo/infer.py --text "Hello world, this is a test using ONNX." --language EN --use_onnx --onnx_path melo_en.onnx --output_dir outputs_onnx
```

**Arguments:**
-   `--use_onnx`: A flag to indicate that an ONNX model should be used for inference.
-   `--onnx_path`: The path to your `.onnx` model file. Required if `--use_onnx` is specified. Defaults to `melo.onnx`.
-   `--language`: The language of the model (must match the language the ONNX model was exported for).
-   `--text`: The text you want to synthesize.
-   `--output_dir`: The directory where the output audio file(s) will be saved.
-   `--config_path`: (Optional) Path to the `config.json` for the model. If not provided, the script will attempt to load/download the default config for the specified language. This is important for `TTS_ONNX` to correctly load model hyperparameters (`hps`).

## Join the Community

**Discord**

Join our [Discord community](https://discord.gg/myshell) and select the `Developer` role upon joining to gain exclusive access to our developer-only channel! Don't miss out on valuable discussions and collaboration opportunities.

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
