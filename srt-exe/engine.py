import argparse
import hashlib
import io
import itertools
import os
from pathlib import Path
import re
import sys
import time

if hasattr(os, "add_dll_directory"):
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    dll_dirs = [path.parent for path in nvidia.rglob("*.dll")]
    dll_dirs = list(dict.fromkeys(dll_dirs))
    dll_handles = [os.add_dll_directory(str(path)) for path in dll_dirs]
    os.environ["PATH"] = os.pathsep.join([*map(str, dll_dirs), os.environ["PATH"]])

import av
import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from filelock import FileLock, Timeout
from opencc import OpenCC
from tokenizers import Tokenizer


RUNTIME = Path(os.environ["LOCALAPPDATA"]) / "Codex" / "windows-chinese-srt"
RATE = 16_000
WINDOW = 240 * RATE
OVERLAP = 2 * RATE
JAPANESE = re.compile(r"[ぁ-ゖァ-ヺｦ-ﾟ]")
KOREAN = re.compile(r"[가-힣]")
HAN = re.compile(r"[㐀-䶿一-鿿]")
LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".ts"}
LANG_CODES = {
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "de": "deu_Latn",
}


def subtitle_language(text, detected):
    if JAPANESE.search(text):
        return "ja"
    if KOREAN.search(text):
        return "ko"
    if HAN.search(text):
        return "zh"
    if not LETTERS.search(text):
        return "zh"
    if detected in LANG_CODES and detected not in {"ja", "ko"}:
        return detected
    return "en"


class ChineseTranslator:
    def __init__(self, models, cpu_threads, device="cpu"):
        path = models / "nllb"
        self.model = ctranslate2.Translator(
            str(path),
            device=device,
            compute_type="int8_float16" if device == "cuda" else "int8",
            intra_threads=cpu_threads,
        )
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.simplify = OpenCC("t2s").convert

    def translate(self, cues):
        texts = [self.simplify(cue[2]) for cue in cues]
        foreign = [(i, cue[2], cue[3]) for i, cue in enumerate(cues) if cue[3] != "zh"]
        for offset in range(0, len(foreign), 32):
            group = foreign[offset : offset + 32]
            encoded = [
                self.tokenizer.encode(value, add_special_tokens=False).tokens
                + ["</s>", LANG_CODES[language]]
                for _, value, language in group
            ]
            results = self.model.translate_batch(
                encoded,
                target_prefix=[["zho_Hans"]] * len(group),
                beam_size=2,
                max_decoding_length=96,
                repetition_penalty=1.1,
            )
            for (index, original, _), result in zip(group, results, strict=True):
                tokens = result.hypotheses[0]
                ids = [self.tokenizer.token_to_id(token) for token in tokens[1:]]
                text = self.tokenizer.decode([token for token in ids if token is not None])
                text = self.simplify(text.strip())
                if not text or (HAN.search(original) is None and text == original):
                    text = "［此处译文待核对］"
                texts[index] = text
        return texts


def stamp(milliseconds):
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def inspect_duration(path):
    with av.open(str(path)) as container:
        return (container.duration or 0) / av.time_base


def media_files(source):
    supported = AUDIO_SUFFIXES | VIDEO_SUFFIXES
    if source.is_file() and source.suffix.lower() in supported:
        return [source]
    if not source.is_dir():
        raise ValueError(f"Expected an audio/video file or folder: {source}")
    pairs = {}
    for path in sorted(source.iterdir(), key=lambda item: item.name.casefold()):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in supported:
            continue
        stem = path.stem.casefold()
        previous = pairs.get(stem)
        if previous is None or (
            suffix in AUDIO_SUFFIXES and previous.suffix.lower() in VIDEO_SUFFIXES
        ):
            pairs[stem] = path
    if not pairs:
        raise ValueError(f"No supported media in {source}")
    return sorted(pairs.values(), key=lambda path: (path.stat().st_size, path.name.casefold()))


def decode_audio(path, limit):
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    samples = io.BytesIO()
    with av.open(str(path)) as container:
        for frame in itertools.chain(container.decode(audio=0), [None]):
            for pcm in resampler.resample(frame):
                samples.write(pcm.to_ndarray().tobytes())
            if limit and samples.tell() >= limit * RATE * 2:
                break
    audio = np.frombuffer(samples.getbuffer(), dtype=np.int16)
    if limit:
        audio = audio[: limit * RATE]
    return audio.astype(np.float32) / 32768.0


def transcribe(path, destination, recognizer, translator, batch_size, limit, gain):
    started = time.monotonic()
    audio = decode_audio(path, limit)
    if gain != 1:
        np.multiply(audio, gain, out=audio)
        np.clip(audio, -1, 1, out=audio)
    cues = []
    prior_end = 0
    for left in range(0, len(audio), WINDOW):
        right = min(left + WINDOW, len(audio))
        expanded_left = max(0, left - OVERLAP)
        expanded_right = min(len(audio), right + OVERLAP)
        segments, info = recognizer.transcribe(
            audio[expanded_left:expanded_right],
            batch_size=batch_size,
            beam_size=1,
            without_timestamps=False,
            multilingual=True,
            language_detection_segments=2,
            no_repeat_ngram_size=5,
            vad_parameters={"threshold": 0.45, "min_silence_duration_ms": 250},
        )
        window_cues = []
        for segment in segments:
            text = " ".join(segment.text.split())
            if not text or not LETTERS.search(text):
                continue
            if segment.no_speech_prob > 0.65 and segment.avg_logprob < -0.9:
                continue
            start_sample = expanded_left + round(segment.start * RATE)
            end_sample = expanded_left + round(segment.end * RATE)
            midpoint = (start_sample + end_sample) // 2
            if not left <= midpoint < right:
                continue
            start = max(prior_end, round(start_sample * 1000 / RATE))
            end = round(min(end_sample, len(audio)) * 1000 / RATE)
            if end <= start:
                continue
            window_cues.append((start, end, text, subtitle_language(text, info.language)))
            prior_end = end
        translations = translator.translate(window_cues)
        for (start, end, _, _), translation in zip(window_cues, translations, strict=True):
            if LETTERS.search(translation):
                cues.append((start, end, " ".join(translation.split())))
        print(
            f"  {right / RATE:.0f}/{len(audio) / RATE:.0f} audio s, "
            f"{len(cues)} cues, {time.monotonic() - started:.1f} elapsed s",
            flush=True,
        )
    if not cues:
        raise ValueError("No speech cues detected")
    partial = destination.with_suffix(".srt.partial")
    with partial.open("w", encoding="utf-8-sig", newline="\r\n") as stream:
        for index, (start, end, text) in enumerate(cues, 1):
            stream.write(f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n")
    try:
        partial.rename(destination)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--models", type=Path, default=RUNTIME / "models")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--translation-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--only", type=int, nargs="+")
    parser.add_argument("--gain", type=float, default=1)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--retry-quiet", action="store_true")
    args = parser.parse_args()
    files = media_files(args.source)
    if args.inspect:
        durations = [inspect_duration(file) for file in files]
        print(f"{len(files)} files, {sum(durations) / 3600:.2f} hours")
        print(f"Shortest: {min(durations):.1f}s. Longest: {max(durations):.1f}s.")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        str(args.models / "whisper-small"),
        device=args.device,
        compute_type="int8_float16" if args.device == "cuda" else "int8",
        cpu_threads=args.cpu_threads,
    )
    recognizer = BatchedInferencePipeline(model)
    translator = ChineseTranslator(args.models, args.cpu_threads, args.translation_device)
    claim_key = hashlib.sha256(str(args.output.resolve()).casefold().encode()).hexdigest()[:20]
    claims = RUNTIME / "locks" / claim_key
    claims.mkdir(parents=True, exist_ok=True)
    failed = 0
    print(f"Ready on {args.device}; {len(files)} files", flush=True)
    jobs = list(enumerate(files, 1))
    if args.only:
        jobs = [job for job in jobs if job[0] in args.only]
    if args.preview:
        jobs = jobs[:1]
    elif args.reverse:
        jobs.reverse()
    for index, path in jobs:
        destination = args.output / path.with_suffix(".srt").name
        if destination.exists():
            print(f"{index}/{len(files)} already finished", flush=True)
            continue
        lock = FileLock(claims / f"{index}.lock", timeout=0)
        try:
            with lock:
                if destination.exists():
                    continue
                print(f"{index}/{len(files)} started  {path.name}", flush=True)
                try:
                    transcribe(
                        path, destination, recognizer, translator,
                        args.batch_size, args.preview, args.gain,
                    )
                except ValueError as error:
                    if not args.retry_quiet or str(error) != "No speech cues detected":
                        raise
                    print("低音量复查中…", flush=True)
                    transcribe(
                        path, destination, recognizer, translator,
                        args.batch_size, args.preview, 8,
                    )
                print(f"{index}/{len(files)} complete", flush=True)
        except Timeout:
            print(f"{index}/{len(files)} assigned to other worker", flush=True)
        except Exception as error:
            print(f"{index}/{len(files)} failed: {error}", flush=True)
            failed += 1
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
