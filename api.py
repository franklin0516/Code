import tempfile
import warnings
from pathlib import Path
from typing import Union
import random 
import numpy as np
from torch import nn

from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.config import load_config


class TTS(nn.Module):
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = "",
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
        gpu=False,
    ):
        """🐸TTS python interface that allows loading and using the released models.

        Example with a multi-speaker model:
            >>> from TTS.api import TTS
            >>> tts = TTS()
            >>> print(tts.list_models().list_models())
            >>> tts_instance = TTS(model_name=tts.list_models().list_models()[0])
        ...

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            progress_bar (bool, optional): Whether to print a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        super().__init__()
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)
        self.config = load_config(config_path) if config_path else None
        self.synthesizer = None
        self.voice_converter = None
        self.model_name = ""
        if gpu:
            warnings.warn("`gpu` will be deprecated. Please use `tts.to(device)` instead.)

        if model_name is not None and len(model_name) > 0:
            if "tts_models" in model_name:
                self.load_tts_model_by_name(model_name, gpu)
            elif "voice_conversion_models" in model_name:
                self.load_vc_model_by_name(model_name, gpu)
            else:
                self.load_model_by_name(model_name, gpu)

        if model_path:
            self.load_tts_model_by_path(
                model_path, config_path, vocoder_path=vocoder_path, vocoder_config=vocoder_config_path, gpu=gpu
            )

    @property
    def models(self):
        return self.manager.list_tts_models()

    @property
    def is_multi_speaker(self):
        if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
            return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_multi_lingual(self):
        if (
            isinstance(self.model_name, str)
            and "xtts" in self.model_name
            or self.config
            and ("xtts" in self.config.model or len(self.config.languages) > 1)
        ):
            return True
        if hasattr(self.synthesizer.tts_model, "language_manager") and self.synthesizer.tts_model.language_manager:
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self):
        if not self.is_multi_speaker:
            return None
        return self.synthesizer.tts_model.speaker_manager.speaker_names

    @property
    def languages(self):
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names

    @staticmethod
    def get_models_file_path():
        return Path(__file__).parent / ".models.json"

    def list_models(self):
        return ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False)

    def download_model_by_name(self, model_name: str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if "fairseq" in model_name or (model_item is not None and isinstance(model_item["model_url"], list)):
            return None, None, None, None, model_path
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path, None

    def load_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.load_tts_model_by_name(model_name, gpu)

    def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the voice conversion models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.model_name = model_name
        model_path, config_path, _, _, _ = self.download_model_by_name(model_name)
        self.voice_converter = Synthesizer(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """
        self.synthesizer = None
        self.model_name = model_name

        model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(
            model_name
        )

        # init synthesizer
        # None values are fetch from the model
        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config_path,
            encoder_checkpoint=None,
            encoder_config=None,
            model_dir=model_dir,
            use_cuda=gpu,
        )

    def load_tts_model_by_path(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        vocoder_checkpoint: str = None,
        vocoder_config: str = None,
        encoder_checkpoint: str = None,
        encoder_config: str = None,
        model_dir: str = None,
        gpu: bool = False,
    ):
        """Load one of 🐸TTS models by path.

        Args:
            tts_checkpoint (str): path to the TTS model checkpoint
            tts_config_path (str): path to the TTS model config file
            vocoder_checkpoint (str, optional): path to the vocoder model checkpoint. Defaults to None.
            vocoder_config (str, optional): path to the vocoder model config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the encoder model checkpoint. Defaults to None.
            encoder_config (str, optional): path to the encoder model config file. Defaults to None.
            model_dir (str, optional): path to the models directory. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.synthesizer = None
        # init synthesizer
        # None values are fetch from the model
        self.synthesizer = Synthesizer(
            tts_checkpoint=tts_checkpoint,
            tts_config_path=tts_config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_checkpoint,
            vocoder_config=vocoder_config,
            encoder_checkpoint=encoder_checkpoint,
            encoder_config=encoder_config,
            model_dir=model_dir,
            use_cuda=gpu,
        )

    def tts(
        self,
        texts: Union[str, list],
        speakers: Union[str, list] = None,
        base_path: str = "tts",
        align: bool = False,
        pitch: float = 0,
        energy: float = 0,
        speed: float = 1,
        out_path: str = None,
    ):
        """Convert text to speech using the loaded TTS model.

        Args:
            texts (Union[str, list]): Input text or list of texts to be synthesized.
            speakers (Union[str, list], optional): Speaker name or list of speaker names for multispeaker models. Defaults to None.
            base_path (str, optional): Base path for saving the generated audio files. Defaults to "tts".
            align (bool, optional): Whether to generate alignment plots. Defaults to False.
            pitch (float, optional): Pitch factor for controlling the pitch of the generated speech. Defaults to 0.
            energy (float, optional): Energy factor for controlling the energy of the generated speech. Defaults to 0.
            speed (float, optional): Speed factor for controlling the speed of the generated speech. Defaults to 1.
            out_path (str, optional): Output file path. If not provided, the audio will be saved with a default name in the base path. Defaults to None.

        Returns:
            Union[str, list]: File path or list of file paths of the generated audio files.
        """
        if not self.synthesizer:
            raise RuntimeError("TTS model not loaded. Use load_model_by_name or load_model_by_path first.")

        if isinstance(texts, str):
            texts = [texts]

        if speakers and not isinstance(speakers, list):
            speakers = [speakers]

        if not speakers and self.is_multi_speaker:
            raise ValueError("Speaker names are required for multispeaker models.")

        out_paths = []

        for i, text in enumerate(texts):
            speaker = speakers[i] if speakers and len(speakers) > i else None
            out_path = self.synthesizer.tts(
                text,
                speaker,
                base_path=base_path,
                align=align,
                pitch=pitch,
                energy=energy,
                speed=speed,
                out_path=out_path,
            )
            out_paths.append(out_path)

        return out_paths


if __name__ == "__main__":
    pass
