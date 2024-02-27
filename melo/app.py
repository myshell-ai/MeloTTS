# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import gradio as gr
import os, torch, io
# os.system('python -m unidic download')
print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")
from melo.api import TTS
speed = 1.0
import tempfile
import click
device = 'auto'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}
speaker_ids = models['EN'].hps.data.spk2id
def synthesize(speaker, text, speed, language, progress=gr.Progress()):
    bio = io.BytesIO()
    models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], bio, speed=speed, pbar=progress.tqdm, format='wav')
    return bio.getvalue()
def load_speakers(language):
    return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys()))
with gr.Blocks() as demo:
    gr.Markdown('# MeloTTS WebUI\n\nA WebUI for MeloTTS.')
    with gr.Group():
        speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-Default', label='Speaker')
        language = gr.Radio(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], label='Language', value='EN')
        language.input(load_speakers, inputs=language, outputs=speaker)
        speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="Text to speak", value='The field of text to speech has seen rapid development recently')
    btn = gr.Button('Synthesize', variant='primary')
    aud = gr.Audio(interactive=False)
    btn.click(synthesize, inputs=[speaker, text, speed, language], outputs=[aud])
    gr.Markdown('WebUI by [mrfakename](https://twitter.com/realmrfakename).')
@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', default=None)
def main(share, host, port):
    demo.queue(api_open=False, default_concurrency_limit=10).launch(show_api=False, share=share, server_name=host, server_port=port)

if __name__ == "__main__":
    main()