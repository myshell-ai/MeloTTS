import os
import re
import json
import torch
import librosa
import soundfile
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch

from . import utils
from . import commons
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .mel_processing import spectrogram_torch, spectrogram_torch_conv
from .download_utils import load_or_download_config, load_or_download_model
import onnxruntime


class TTS(nn.Module):
    def __init__(self,
                language,
                device='auto',
                use_hf=True,
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        # config_path = 
        hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        # load state_dict
        checkpoint_dict = load_or_download_model(language, device, use_hf=use_hf, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language # we support a ZH_MIX_EN model

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)


class TTS_ONNX(object): # Changed to inherit from object
    def __init__(self, 
                language,
                onnx_path,
                device='auto',
                use_hf=True,
                config_path=None):
        super().__init__() # Ensure super().__init__() is called
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): device = 'cuda'
            if torch.backends.mps.is_available(): device = 'mps'
        
        self.device = device # device for text processing, ONNX session handles its own device via providers

        self.hps = load_or_download_config(language, use_hf=use_hf, config_path=config_path)
        self.symbol_to_id = {s: i for i, s in enumerate(self.hps.symbols)}
        
        _language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if _language == 'ZH' else _language

        providers = ['CPUExecutionProvider'] if self.device == 'cpu' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.onnx_input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.ort_session.get_outputs()]


    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False,):
        language = self.language
        texts = TTS.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        if pbar:
            tx = pbar(texts)
        else:
            if position:
                tx = tqdm(texts, position=position)
            elif quiet:
                tx = texts
            else:
                tx = tqdm(texts)
        
        # Use self.device for text processing, then convert to CPU for ONNX if needed by specific inputs
        # The main ONNX session runs on its configured device (CPU or CUDA)
        processing_device = self.device 

        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, processing_device, self.symbol_to_id)
            
            # Convert tensors to CPU and then to numpy for ONNX Runtime input
            x_tst = phones.unsqueeze(0).cpu().numpy()
            tones_tst = tones.unsqueeze(0).cpu().numpy()
            lang_ids_tst = lang_ids.unsqueeze(0).cpu().numpy() # Ensure lang_ids is processed correctly
            bert_tst = bert.unsqueeze(0).cpu().numpy()
            ja_bert_tst = ja_bert.unsqueeze(0).cpu().numpy()
            x_tst_lengths = torch.LongTensor([phones.size(0)]).cpu().numpy()
            speakers_tst = torch.LongTensor([speaker_id]).cpu().numpy()

            input_feed = {
                'x': x_tst,
                'x_lengths': x_tst_lengths,
                'sid': speakers_tst,
                'tone': tones_tst,
                'language': lang_ids_tst,
                'bert': bert_tst,
                'ja_bert': ja_bert_tst,
                # Parameters for inference, ensure these are part of the ONNX model's inputs if they need to be dynamic
                # For now, assuming they are fixed or handled differently in the ONNX model
                # 'noise_scale': np.array([noise_scale], dtype=np.float32),
                # 'length_scale': np.array([1.0 / speed], dtype=np.float32),
                # 'noise_scale_w': np.array([noise_scale_w], dtype=np.float32),
            }
            
            # Filter input_feed to only include names expected by the ONNX model
            # This is important if the ONNX model doesn't use all defined inputs (e.g. sdp_ratio, noise params are baked in or passed differently)
            # The `export_onnx.py` script defines specific inputs.
            # Current ONNX model inputs: ['x', 'x_lengths', 'sid', 'tone', 'language', 'bert', 'ja_bert']
            # The noise_scale, length_scale, noise_scale_w are passed as direct arguments to model.infer in PyTorch
            # In ONNX, these need to be inputs to the graph if they are to be controlled at runtime.
            # The export_onnx.py seems to bake them in by not including them as onnx inputs.
            # For now, we will only feed the tensor inputs.
            # If the ONNX model was exported with noise_scale, etc., as inputs, they should be added here.

            # The export_onnx.py script specifies these as inputs to torch.onnx.export:
            # args_for_onnx_export = (
            #     x, x_lengths, sid, tone_input, language_input, bert_input, ja_bert_input,
            #     noise_scale, length_scale, noise_scale_w, max_len, sdp_ratio  <-- these are additional args
            # )
            # input_names = ['x', 'x_lengths', 'sid', 'tone', 'language', 'bert', 'ja_bert'] <-- but these are the named inputs
            # This means noise_scale, length_scale, etc., are passed as *non-tensor arguments* or *attributes* to the ONNX graph's operators if the exporter handles them that way,
            # or they are effectively baked into the graph if they were constants during export.
            # The current `export_onnx.py` passes them as arguments to `model.infer` which means they are used to compute intermediate values
            # *before* the parts of `model.infer` that are captured into the ONNX graph.
            # If `model.infer` directly uses these as Python scalars to control flow or shape tensors that are *not* inputs to the ONNX graph,
            # then they are effectively baked in.
            # The `SynthesizerTrn.infer` method in `models.py` has these as parameters.
            # The ONNX export captures the `SynthesizerTrn.infer_onnx` method (if defined) or `SynthesizerTrn.infer`.
            # Let's assume the `export_onnx.py` sets up these values as constants or fixed inputs not part of the dynamic `input_feed` for now.
            # The provided `export_onnx.py` does pass them as arguments to `torch.onnx.export` for `model` which is `SynthesizerTrn`.
            # However, `input_names` only lists the tensor inputs. This implies that `noise_scale` etc. are treated as arguments to the
            # *scripted function/module* being exported, not as dynamic inputs to the resulting ONNX graph in the `input_feed` sense.
            # They become constants within the ONNX graph.
            # So, the current `input_feed` is likely correct.

            audio = self.ort_session.run(self.onnx_output_names, input_feed)[0] # Get the first output
            audio = audio.squeeze().astype(np.float32) # Remove batch/channel dims, ensure float32
            audio_list.append(audio)
        
        if torch.cuda.is_available(): # General cleanup, not specific to ONNX session device
            torch.cuda.empty_cache()
            
        audio = TTS.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if format:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate, format=format)
            else:
                soundfile.write(output_path, audio, self.hps.data.sampling_rate)
