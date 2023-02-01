import numpy as np
import torch
import whisper
import os
import base64
from io import BytesIO
from pydub import AudioSegment
from pyannote.audio import Pipeline
from sanic.log import logger as log

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, pipeline
    
    model = whisper.load_model("medium")
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token=True)

def info():
    return {
        'cuda': {
            'is_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
            'segmentation': all(p.is_cuda for p in pipeline._segmentation.model.parameters()),
            'classifier': all(p.is_cuda for p in pipeline._embedding.classifier_.parameters()),
            'asr': all(p.is_cuda for p in model.parameters()),
        },
        'device': {
            'segmentation': str(pipeline._segmentation.model.device),
            'classifier': str(pipeline._embedding.classifier_.device),
            'asr': str(model.device),
        },
    }

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model, pipeline

    info_val = model_inputs.get('info')
    if info_val:
        return dict(info=info())

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    del model_inputs
    if mp3BytesString == None:
        return {'message': "No input provided"}
    
    audio_file = base64.b64decode(mp3BytesString)
    del mp3BytesString

    log.warn(f'Inference request received, audio file {len(audio_file)} bytes');
    with BytesIO(audio_file) as mp3Bytes:
        del audio_file
        with open('input.ogg','wb') as file:
            file.write(mp3Bytes.getbuffer())

    original_audio = AudioSegment.from_file('input.ogg')
    log.warn(f'Reading audio file. Sample rate: {original_audio.frame_rate}Hz. Channels: {original_audio.channels}. Duration: {len(original_audio)}ms');
    audio = original_audio.split_to_mono()[0].set_frame_rate(16000)
    del original_audio
    log.warn(f'Diarizing...')
    dz = pipeline('input.ogg')
    output = []
    previous_ts = 0
    for turn, _, speaker in dz.itertracks(yield_label=True):
        log.warn(f'ASR on segment {turn.start}-{turn.end}...')
        start, end = int(turn.start * 1000), int(turn.end * 1000)
        ts0 = min(max(start - 500, previous_ts), start)
        segment = audio[ts0:end]
        samples = segment.get_array_of_samples()
        array = np.array(samples).astype(np.float32)
        array /= np.iinfo(samples.typecode).max
        transcription = model.transcribe(array, language='en')
        output.append(dict(start=turn.start, end=turn.end, speaker=speaker, transcription=transcription.get('text')))
        previous_ts = end

    # Run the model
    #result = model.transcribe("input.ogg")
    #output = dict(text=result['text'], model='medium')
    #output = dict(dz=[dict(turn=turn, speaker=speaker) for turn, _, speaker in dz.itertracks(yield_label=True)])
    log.warn(f'Returning results')
    os.remove("input.ogg")
    # Return the results as a dictionary
    if info_val == False:
        return dict(transcription=output)
    else:
        return dict(transcription=output, info=info())
