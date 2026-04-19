"""Clean Kokoro implementation with controlled resource management."""

import asyncio
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import AsyncGenerator, Dict, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KModel, KPipeline
from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import model_config
from ..structures.schemas import WordTimestamp
from .base import AudioChunk, BaseModelBackend


class KokoroV1(BaseModelBackend):
    """Kokoro backend with controlled resource management."""

    def __init__(self):
        """Initialize backend with environment-based configuration."""
        super().__init__()
        # Strictly respect settings.use_gpu
        self._device = settings.get_device()
        self._torch_device = torch.device(self._device)
        self._is_cuda = self._torch_device.type == "cuda"
        self._amp_dtype = self._resolve_amp_dtype(settings.amp_dtype)
        if self._is_cuda:
            torch.cuda.set_device(self._torch_device)
        self._model: Optional[KModel] = None
        self._pipelines: Dict[str, KPipeline] = {}  # Store pipelines by lang_code
        self._pipeline_lock = threading.Lock()
        self._rnn_flatten_lock = threading.Lock()
        self._worker_pipelines: Dict[Tuple[int, str], KPipeline] = {}
        self._voice_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._voice_cache_lock = threading.Lock()
        self._thread_local = threading.local()
        self._workers = max(1, settings.gpu_inference_workers)
        self._executor = ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="kokoro-infer"
        )
        self._inference_semaphore = asyncio.Semaphore(
            max(1, settings.gpu_inference_concurrency)
        )

    async def load_model(self, path: str) -> None:
        """Load pre-baked model.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Get verified model path
            model_path = await paths.get_model_path(path)
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

            if not os.path.exists(config_path):
                raise RuntimeError(f"Config file not found: {config_path}")

            logger.info(f"Loading Kokoro model on {self._device}")
            logger.info(f"Config path: {config_path}")
            logger.info(f"Model path: {model_path}")

            # Load model and pin it to the configured device. This supports
            # explicit CUDA devices such as "cuda:0", not only bare "cuda".
            self._model = KModel(config=config_path, model=model_path).eval()
            if self._torch_device.type == "mps":
                logger.info("Moving model to MPS device")
            self._model = self._model.to(self._torch_device)
            self._flatten_rnn_parameters()

        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load Kokoro model: {e}")

    @staticmethod
    def _resolve_amp_dtype(dtype_name: str):
        """Map config strings to torch dtypes used by autocast."""
        normalized = dtype_name.lower().strip()
        if normalized == "fp32":
            return None
        if normalized == "fp16":
            return torch.float16
        if normalized == "bf16":
            return torch.bfloat16
        raise ValueError(
            f"Unsupported AMP_DTYPE={dtype_name!r}; expected fp32, fp16, or bf16"
        )

    def _inference_context(self):
        """Return the PyTorch context for model inference."""
        grad_context = (
            torch.inference_mode() if settings.use_inference_mode else torch.no_grad()
        )
        autocast_context = (
            torch.amp.autocast("cuda", dtype=self._amp_dtype)
            if self._is_cuda and self._amp_dtype is not None
            else nullcontext()
        )
        return grad_context, autocast_context

    def _flatten_rnn_parameters(self, log: bool = True) -> None:
        """Keep RNN weights contiguous to avoid per-call cuDNN repacking."""
        if not self._model:
            return

        flattened = 0
        with self._rnn_flatten_lock:
            for module in self._model.modules():
                flatten = getattr(module, "flatten_parameters", None)
                if flatten is not None:
                    flatten()
                    flattened += 1

        if flattened and log:
            logger.info(f"Flattened parameters for {flattened} recurrent modules")

    async def warmup_workers(self, lang_code: str) -> None:
        """Create per-thread G2P pipelines before serving concurrent traffic."""
        if not self.is_loaded or self._workers <= 1:
            return

        barrier = threading.Barrier(self._workers)

        def warmup_worker():
            barrier.wait()
            self._get_worker_pipeline(lang_code)
            # The warmup request can invalidate cuDNN's packed RNN layout later in
            # startup, so force one cheap flatten on the first real request.
            self._thread_local.rnn_flattened = False

        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *[
                loop.run_in_executor(self._executor, warmup_worker)
                for _ in range(self._workers)
            ]
        )

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        """Get or create pipeline for language code.

        Args:
            lang_code: Language code to use

        Returns:
            KPipeline instance for the language
        """
        if not self._model:
            raise RuntimeError("Model not loaded")

        if lang_code not in self._pipelines:
            with self._pipeline_lock:
                if lang_code not in self._pipelines:
                    logger.info(f"Creating new pipeline for language code: {lang_code}")
                    self._pipelines[lang_code] = KPipeline(
                        lang_code=lang_code, model=self._model, device=self._device
                    )
                    self._flatten_rnn_parameters()
                    self._thread_local.rnn_flattened = False
        return self._pipelines[lang_code]

    def _get_worker_pipeline(self, lang_code: str) -> KPipeline:
        """Get a per-worker text pipeline so concurrent G2P state is isolated."""
        key = (threading.get_ident(), lang_code)
        pipeline = self._worker_pipelines.get(key)
        if pipeline is None:
            with self._pipeline_lock:
                pipeline = self._worker_pipelines.get(key)
                if pipeline is None:
                    logger.info(
                        f"Creating worker pipeline for language code: {lang_code}"
                    )
                    pipeline = KPipeline(
                        lang_code=lang_code, model=False, device=self._device
                    )
                    self._worker_pipelines[key] = pipeline
                    self._flatten_rnn_parameters()
                    self._thread_local.rnn_flattened = False
        return pipeline

    def _get_cuda_stream(self):
        """Return a CUDA stream local to each inference worker."""
        if not self._is_cuda or not settings.gpu_use_streams:
            return None
        stream = getattr(self._thread_local, "cuda_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self._torch_device)
            self._thread_local.cuda_stream = stream
        return stream

    def _load_voice_pack(self, voice_path: str) -> torch.Tensor:
        """Load and cache a voice pack on the target device."""
        if not model_config.cache_voices:
            return torch.load(voice_path, weights_only=True).to(
                self._torch_device, non_blocking=True
            )

        with self._voice_cache_lock:
            cached = self._voice_cache.get(voice_path)
            if cached is not None:
                self._voice_cache.move_to_end(voice_path)
                return cached

        pack = torch.load(voice_path, weights_only=True).to(
            self._torch_device, non_blocking=True
        )

        with self._voice_cache_lock:
            cached = self._voice_cache.get(voice_path)
            if cached is not None:
                self._voice_cache.move_to_end(voice_path)
                return cached
            self._voice_cache[voice_path] = pack
            max_size = max(1, model_config.voice_cache_size)
            while len(self._voice_cache) > max_size:
                self._voice_cache.popitem(last=False)
        return pack

    def _resolve_voice_path(
        self, voice: Union[str, Tuple[str, Union[torch.Tensor, str]]]
    ) -> Tuple[str, str]:
        """Normalize voice input to a name and path usable by Kokoro."""
        if isinstance(voice, tuple):
            voice_name, voice_data = voice
            if isinstance(voice_data, str):
                return voice_name, voice_data

            import tempfile

            voice_path = os.path.join(tempfile.gettempdir(), f"{voice_name}.pt")
            torch.save(voice_data.cpu(), voice_path)
            return voice_name, voice_path

        return os.path.splitext(os.path.basename(voice))[0], voice

    def _pipeline_lang_code(self, voice_name: str, lang_code: Optional[str]) -> str:
        if lang_code:
            return lang_code
        if settings.default_voice_code:
            return settings.default_voice_code
        return voice_name[0].lower()

    def _iter_text_segments(self, pipeline: KPipeline, text: str):
        """Tokenize text like KPipeline.__call__, but leave inference to us."""
        grapheme_chunks = re.split(r"\n+", text.strip()) if text.strip() else []
        for graphemes_index, graphemes in enumerate(grapheme_chunks):
            if not graphemes.strip():
                continue

            if pipeline.lang_code in "ab":
                logger.debug(
                    f"Processing English text: {graphemes[:50]}{'...' if len(graphemes) > 50 else ''}"
                )
                _, tokens = pipeline.g2p(graphemes)
                for gs, ps, tks in pipeline.en_tokenize(tokens):
                    if not ps:
                        continue
                    if len(ps) > 510:
                        logger.warning(
                            f"Unexpected len(ps) == {len(ps)} > 510 and ps == '{ps}'"
                        )
                        ps = ps[:510]
                    yield gs, ps, tks, graphemes_index
                continue

            chunk_size = 400
            chunks = []
            sentences = re.split(r"([.!?]+)", graphemes)
            current_chunk = ""

            for i in range(0, len(sentences), 2):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]

                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            if not chunks:
                chunks = [
                    graphemes[i : i + chunk_size]
                    for i in range(0, len(graphemes), chunk_size)
                ]

            for chunk in chunks:
                if not chunk.strip():
                    continue
                ps, _ = pipeline.g2p(chunk)
                if not ps:
                    continue
                if len(ps) > 510:
                    logger.warning(f"Truncating len(ps) == {len(ps)} > 510")
                    ps = ps[:510]
                yield chunk, ps, None, graphemes_index

    def _timestamps_from_tokens(self, tokens, pred_dur) -> Optional[list[WordTimestamp]]:
        if not tokens or pred_dur is None:
            return None

        word_timestamps = []
        try:
            KPipeline.join_timestamps(tokens, pred_dur)
            for token in tokens:
                if not all(
                    hasattr(token, attr) for attr in ["text", "start_ts", "end_ts"]
                ):
                    continue
                if not token.text or not token.text.strip():
                    continue
                word_timestamps.append(
                    WordTimestamp(
                        word=str(token.text).strip(),
                        start_time=float(token.start_ts),
                        end_time=float(token.end_ts),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to process timestamps for chunk: {e}")

        return word_timestamps or None

    def _infer_phonemes_sync(
        self, phonemes: str, pack: torch.Tensor, speed: float
    ) -> KModel.Output:
        """Infer one phoneme string using the configured Kokoro model."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        if self._is_cuda and not getattr(self._thread_local, "rnn_flattened", False):
            self._flatten_rnn_parameters(log=False)
            self._thread_local.rnn_flattened = True
        return KPipeline.infer(self._model, phonemes, pack, speed)

    def _generate_text_sync(
        self,
        text: str,
        voice_path: str,
        speed: float,
        pipeline_lang_code: str,
        return_timestamps: bool,
    ) -> list[AudioChunk]:
        if not self._model:
            raise RuntimeError("Model not loaded")

        pipeline = self._get_worker_pipeline(pipeline_lang_code)
        pack = self._load_voice_pack(voice_path)
        stream = self._get_cuda_stream()
        stream_context = (
            torch.cuda.stream(stream) if stream is not None else nullcontext()
        )
        chunks = []

        grad_context, autocast_context = self._inference_context()
        with grad_context, autocast_context, stream_context:
            for _, phonemes, tokens, _ in self._iter_text_segments(pipeline, text):
                output = self._infer_phonemes_sync(phonemes, pack, speed)
                if output.audio is None:
                    logger.warning("No audio in chunk")
                    continue

                logger.debug(f"Got audio chunk with shape: {output.audio.shape}")
                word_timestamps = (
                    self._timestamps_from_tokens(tokens, output.pred_dur)
                    if return_timestamps
                    else None
                )
                chunks.append(
                    AudioChunk(output.audio.numpy(), word_timestamps=word_timestamps)
                )

        return chunks

    def _generate_tokens_sync(
        self,
        tokens: str,
        voice_path: str,
        speed: float,
        pipeline_lang_code: str,
    ) -> list[np.ndarray]:
        if not self._model:
            raise RuntimeError("Model not loaded")

        pack = self._load_voice_pack(voice_path)
        stream = self._get_cuda_stream()
        stream_context = (
            torch.cuda.stream(stream) if stream is not None else nullcontext()
        )
        chunks = []

        grad_context, autocast_context = self._inference_context()
        with grad_context, autocast_context, stream_context:
            if len(tokens) > 510:
                raise ValueError(f"Phoneme string too long: {len(tokens)} > 510")
            output = self._infer_phonemes_sync(tokens, pack, speed)
            if output.audio is not None:
                chunks.append(output.audio.numpy())

        return chunks

    async def _run_inference(self, func, *args):
        async with self._inference_semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, func, *args)

    async def generate_from_tokens(
        self,
        tokens: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate audio from phoneme tokens.

        Args:
            tokens: Input phoneme tokens to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        try:
            if self._is_cuda:
                if self._check_memory():
                    self._clear_memory()

            voice_name, voice_path = self._resolve_voice_path(voice)
            pipeline_lang_code = self._pipeline_lang_code(voice_name, lang_code)

            logger.debug(
                f"Generating audio from tokens with lang_code '{pipeline_lang_code}': '{tokens[:100]}{'...' if len(tokens) > 100 else ''}'"
            )
            chunks = await self._run_inference(
                self._generate_tokens_sync,
                tokens,
                voice_path,
                speed,
                pipeline_lang_code,
            )
            for chunk in chunks:
                yield chunk

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._is_cuda
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate_from_tokens(
                    tokens, voice, speed, lang_code
                ):
                    yield chunk
            raise

    async def generate(
        self,
        text: str,
        voice: Union[str, Tuple[str, Union[torch.Tensor, str]]],
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate audio using model.

        Args:
            text: Input text to synthesize
            voice: Either a voice path string or a tuple of (voice_name, voice_tensor/path)
            speed: Speed multiplier
            lang_code: Optional language code override

        Yields:
            Generated audio chunks

        Raises:
            RuntimeError: If generation fails
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        try:
            if self._is_cuda:
                if self._check_memory():
                    self._clear_memory()

            voice_name, voice_path = self._resolve_voice_path(voice)
            pipeline_lang_code = self._pipeline_lang_code(voice_name, lang_code)

            logger.debug(
                f"Generating audio for text with lang_code '{pipeline_lang_code}': '{text[:100]}{'...' if len(text) > 100 else ''}'"
            )
            chunks = await self._run_inference(
                self._generate_text_sync,
                text,
                voice_path,
                speed,
                pipeline_lang_code,
                bool(return_timestamps),
            )
            for chunk in chunks:
                yield chunk

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            if (
                self._is_cuda
                and model_config.pytorch_gpu.retry_on_oom
                and "out of memory" in str(e).lower()
            ):
                self._clear_memory()
                async for chunk in self.generate(text, voice, speed, lang_code):
                    yield chunk
            raise

    def _check_memory(self) -> bool:
        """Check if memory usage is above threshold."""
        if self._is_cuda:
            threshold = model_config.pytorch_gpu.memory_threshold
            memory_allocated = torch.cuda.memory_allocated(self._torch_device)
            if threshold <= 1:
                total_memory = torch.cuda.get_device_properties(
                    self._torch_device
                ).total_memory
                return (memory_allocated / total_memory) > threshold
            return (memory_allocated / 1e9) > threshold
        # MPS doesn't provide memory management APIs
        return False

    def _clear_memory(self) -> None:
        """Clear device memory."""
        if self._is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self._torch_device)
        elif self._device == "mps":
            # Empty cache if available (future-proofing)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        for pipeline in self._pipelines.values():
            del pipeline
        self._pipelines.clear()
        for pipeline in self._worker_pipelines.values():
            del pipeline
        self._worker_pipelines.clear()
        with self._voice_cache_lock:
            self._voice_cache.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self._is_cuda:
                torch.cuda.synchronize(self._torch_device)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._device
