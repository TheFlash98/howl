import logging
import time
from typing import Callable

import numpy as np
import pyaudio
import torch

from howl.context import InferenceContext
from howl.model.inference import InferenceEngine
import time


class HowlClient2:
    """
    A client for serving Howl models. Users can provide custom listener callbacks
    when a wake word is detected using the `add_listener` method.

    The `from_pretrained` method allows users to directly load a pretrained model by name,
    such as "hey_fire_fox", if the model is not provided upon initialization.
    """

    def __init__(self,
                 engine: InferenceEngine = None,
                 engine2: InferenceEngine = None,
                 context: InferenceContext = None,
                 context2: InferenceContext = None,
                 device: int = -1,
                 chunk_size: int = 500):
        """
        Initializes the client.

        Parameters:
            engine (InferenceEngine): An InferenceEngine for processing audio input, provided
                                        on initialization or by loading a pretrained model.
            context (InferenceContext): An InferenceContext object containing vocab, provided
                                        on initialization or by loading a pretrained model.
            device (int): Device for CPU/GPU support. Defaults to -1 for running on CPU,
                            positive values will run the model on the associated CUDA device.
            chunk_size (int): Number of frames per buffer for audio.
        """
        self.listeners = []
        self.chunk_size = chunk_size
        self.device = self._get_device(device)

        self.engine: InferenceEngine = engine
        self.engine2 : InferenceEngine = engine2
        self.ctx: InferenceContext = context
        self.ctx2: InferenceContext = context2

        self._audio_buf = []
        self._audio_buf_len = 16
        self._audio_float_size = 32767
        self._infer_detected = False
        self._infer_detected_2 = False
        self.last_data = np.zeros(self.chunk_size)

        self.hey_timestamp = []
        self.augnito_timestamp = []

    @staticmethod
    def list_pretrained(force_reload: bool = False):
        """Show a list of available pretrained models"""
        print(torch.hub.list('castorini/howl', force_reload=force_reload))

    @staticmethod
    def _get_device(device: int):
        if device == -1:
            return torch.device('cpu')
        else:
            return torch.device('cuda:{}'.format(device))

    def _on_audio(self, in_data, frame_count, time_info, status):
        data_ok = (in_data, pyaudio.paContinue)
        self.last_data = in_data
        self._audio_buf.append(in_data)
        if len(self._audio_buf) != self._audio_buf_len:
            return data_ok

        audio_data = b''.join(self._audio_buf)
        self._audio_buf = self._audio_buf[2:]
        arr = self._normalize_audio(audio_data)
        inp = torch.from_numpy(arr).float().to(self.device)
        # Inference from input sequence
        if self.engine.infer(inp):
            # Check if inference has already occured for this sequence to prevent
            # duplicate callback execution
            if self._infer_detected:
                return data_ok
            #print(self.engine.sequence)

            self._infer_detected = True
            # print(self.engine2.label_history)
            # print(self.engine.label_history)
            # augnito_history = self.engine.label_history[-1]
            # print(augnito_history)
            augnito_timestamp = time.time()
            hey_timestamp = self.hey_timestamp[-1]
            # for history in self.engine2.label_history:
            #     hey_label = history[1]
            #     if hey_label == 0:
            #         hey_timestamp = history[0]
            print(augnito_timestamp - hey_timestamp)
            if(((augnito_timestamp - hey_timestamp) < 3 and (augnito_timestamp - hey_timestamp) > 0) or self.engine.prediction_confidence > 0.80):
                print("WAKE UP")


            phrase = ' '.join(self.ctx.vocab[x]
                              for x in self.engine.sequence).title()
            # print(self.engine.sequence, "engine 1")
            #phrase = self.ctx.vocab[self.engine.detected_label]
            prediction_confidence = self.engine.prediction_confidence
            logging.info(f'{phrase} detected with {prediction_confidence} confidence by engine 1')
            # Execute user-provided listener callbacks
            for lis in self.listeners:
                lis(self.engine.sequence)
        else:
            self._infer_detected = False
        
        # Inference from input sequence using second engine
        if self.engine2.infer(inp):
            # Check if inference has already occured for this sequence to prevent
            # duplicate callback execution
            if self._infer_detected_2:
                return data_ok
            # print(self.engine2.sequence, "engine 2")
            # print(self.engine2.label_history)
            self._infer_detected_2 = True
            if(len(self.hey_timestamp) > 20):
                self.hey_timestamp = []
                self.hey_timestamp.append(time.time())
            else:
                self.hey_timestamp.append(time.time())
            phrase = ' '.join(self.ctx2.vocab[x]
                             for x in self.engine2.sequence).title()
            prediction_confidence = self.engine2.prediction_confidence
            logging.info(f'{phrase} detected with {prediction_confidence} confidence by engine 2')
            for lis in self.listeners:
                lis(self.engine2.sequence)
        else:
            self._infer_detected_2 = False
        return data_ok

    def _normalize_audio(self, audio_data):
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float) / self._audio_float_size

    def start(self):
        """Start the audio stream for inference"""
        if self.engine is None:
            raise AttributeError(
                'Please provide an InferenceEngine or initialize using from_pretrained.'
            )
        if self.ctx is None:
            raise AttributeError(
                'Please provide an InferenceContext or initialize using from_pretrained.'
            )

        chosen_idx = 0
        self._audio = pyaudio.PyAudio()
        for idx in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(idx)
            if info['name'] == 'pulse':
                chosen_idx = idx
                break

        stream = self._audio.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  input_device_index=chosen_idx,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self._on_audio)
        self.stream = stream
        logging.info("Starting Howl inference client...")
        stream.start_stream()
        return self

    def join(self):
        """Block while the audio inference stream is active"""
        while self.stream.is_active():
            time.sleep(0.1)

    def from_pretrained(self, name: str, force_reload: bool = False):
        """Load a pretrained model using the provided name"""
        engine, ctx = torch.hub.load(
            'castorini/howl', name, force_reload=force_reload, reload_models=force_reload, device=self.device)
        self.engine = engine.to(self.device)
        self.ctx = ctx

    def add_listener(self, listener: Callable):
        """
        Add a listener callback to be executed when a sequence is detected

        Parameters:
            listener (Callable): A function that takes in a list of strings as an argument,
                                 which will contain the detected wake words.
        """
        self.listeners.append(listener)
