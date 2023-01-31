import numpy as np
import torch
import whisper
import os
import base64
from io import BytesIO
from pydub import AudioSegment
from pyannote.audio import Pipeline

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
    
    with BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1"))) as mp3Bytes:
        del mp3BytesString
        with open('input.ogg','wb') as file:
            file.write(mp3Bytes.getbuffer())

    audio = AudioSegment.from_file('input.ogg').split_to_mono()[0].set_frame_rate(16000)
    dz = pipeline('input.ogg')
    output = []
    previous_ts = 0
    for turn, _, speaker in dz.itertracks(yield_label=True):
        start, end = int(turn.start * 1000), int(turn.end * 1000)
        ts0 = min(max(start - 500, previous_ts), start)
        segment = audio[ts0:end]
        samples = segment.get_array_of_samples()
        array = np.array(samples).astype(np.float32)
        array /= np.iinfo(samples.typecode).max
        transcription = model.transcribe(array, language='en')
        output.append(dict(start=turn.start, end=turn.end, speaker=speaker, transcription=transcription['text']))
        previous_ts = end

    # Run the model
    #result = model.transcribe("input.ogg")
    #output = dict(text=result['text'], model='medium')
    #output = dict(dz=[dict(turn=turn, speaker=speaker) for turn, _, speaker in dz.itertracks(yield_label=True)])
    os.remove("input.ogg")
    # Return the results as a dictionary
    if info_val == False:
        return dict(transcription=output)
    else:
        return dict(transcription=output, info=info())
