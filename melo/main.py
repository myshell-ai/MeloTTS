import click
import warnings

@click.command
@click.argument('text')
@click.argument('output_path')
@click.option('--language', '-l', default='EN', help='Language, defaults to English', type=click.Choice(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], case_sensitive=False))
@click.option('--speaker', '-spk', default='EN-Default', help='Speaker ID, only for English, leave empty for default, ignored if not English. If English, defaults to "EN-Default"', type=click.Choice(['EN-Default', 'EN-US', 'EN-BR', 'EN-INDIA', 'EN-AU']))
@click.option('--speed', '-s', default=1.0, help='Speed, defaults to 1.0', type=float)
@click.option('--device', '-d', default='auto', help='Device, defaults to auto')
def main(text, output_path, language, speaker, speed, device):
    language = language.upper()
    if language == '': language = 'EN'
    if speaker == '': speaker = None
    if (not language == 'EN') and speaker:
        warnings.warn('You specified a speaker but the language is English.')
    from melo.api import TTS
    model = TTS(language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if language == 'EN':
        if not speaker: speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    model.tts_to_file(text, spkr, output_path, speed=speed)

if __name__ == "__main__":
    main()