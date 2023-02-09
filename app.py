import time
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
    # pipeline._segmentation.progress_hook = print

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


class Stopwatch(object):
    def __init__(self):
        self._ts0 = time.time()
        self._ts = self._ts0
        self._periods = {}

    def lapse(self, label):
        since = self._ts
        self._ts = time.time()
        self._periods.setdefault(label, []).append(self._ts - since)

    def report(self):
        def f(n):
            return int(n * 1000)
        def summarize(n):
            if len(n) == 1:
                return f(n[0])
            arr = np.array(n)
            return dict(
                total=f(arr.sum()),
                count=arr.size,
                mean=f(arr.mean()),
                sd=f(arr.std()),
                max=f(arr.max()),
                min=f(arr.min()),
                pct99=f(np.percentile(arr, 99)),
                pct90=f(np.percentile(arr, 90)),
            )

        total = f(time.time() - self._ts0)
        lapses = {l: summarize(n) for l, n in self._periods.items()}
        return dict(total=total, lapses=lapses)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model, pipeline
    sw = Stopwatch()

    info_val = model_inputs.get('info')
    if info_val == 'now':
        return dict(info=info())

    # Parse out your arguments
    mp3BytesString = model_inputs.get('mp3BytesString', None)
    del model_inputs
    if mp3BytesString == None:
        return {'message': "No input provided"}

    audio_file = base64.b64decode(mp3BytesString)
    del mp3BytesString

    log.info(f'Inference request received, audio file {len(audio_file)} bytes');
    with BytesIO(audio_file) as mp3Bytes:
        del audio_file
        with open('input.ogg','wb') as file:
            file.write(mp3Bytes.getbuffer())

    sw.lapse('write')

    waveform = AudioSegment.from_file('input.ogg')
    os.remove("input.ogg")
    log.info(f'Reading audio file. Sample rate: {waveform.frame_rate}Hz. Channels: {waveform.channels}. Duration: {len(waveform)}ms');
    waveform = waveform.split_to_mono()[0].set_frame_rate(16000)
    waveform = waveform.get_array_of_samples()
    waveform = torch.tensor(waveform, dtype=torch.float32).reshape(1, -1) / np.iinfo(waveform.typecode).max
    sw.lapse('load')

    log.info(f'Diarizing...')
    dz = pipeline(dict(waveform=waveform, sample_rate=16000))
    sw.lapse('diarization')

    waveform = waveform.reshape(-1)
    output = []
    previous_ts = 0
    for turn, _, speaker in dz.itertracks(yield_label=True):
        log.info(f'ASR on segment {turn.start:.3f}-{turn.end:.3f}...')
        start, end = int(turn.start * 16000), int(turn.end * 16000)
        ts0 = min(max(start - 8000, previous_ts), start)
        segment = waveform[ts0:end]
        transcription = model.transcribe(segment, language='en')
        output.append(dict(start=start / 16000, end=end / 16000, speaker=speaker, transcription=transcription.get('text')))
        previous_ts = end
        sw.lapse('transcription')

    log.info(f'Returning results')
    # Return the results as a dictionary
    rv = dict(transcription=output, timings=sw.report())
    if info_val:
        rv['info'] = info()
    return rv
