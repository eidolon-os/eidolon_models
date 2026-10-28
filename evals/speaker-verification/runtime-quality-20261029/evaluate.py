"""Exercise the actual Channel VAD/embedding adapter on pinned public audio."""
import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

from eidolon.livekit.plugins.speaker_verification.reference_extractor import (
    InvalidSpeakerAudio, ReferenceSpeakerExtractor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = json.loads(args.assets.read_text())
    selected = {}
    for row in assets['samples']:
        selected.setdefault(row['speaker'], row)
    extractor = ReferenceSpeakerExtractor(args.model)
    started = time.perf_counter()
    extractor.warm_up()
    warmup = time.perf_counter() - started
    results = []
    for row in selected.values():
        path = args.assets.parent / row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            pcm = wav.readframes(wav.getnframes())
        started = time.perf_counter()
        try:
            sample = extractor.extract(pcm, min_speech_ms=1500)
            outcome = dict(status='extracted', quality=sample.quality, vector_dimension=len(sample.vector))
        except InvalidSpeakerAudio as exc:
            outcome = dict(status='rejected', reason=str(exc),
                           quality=exc.quality.model_dump(exclude_none=True) if exc.quality else None)
        results.append(dict(sample_id=row['id'], sha256=row['sha256'],
                            elapsed_ms=(time.perf_counter()-started)*1000, **outcome))
    args.output.write_text(json.dumps(dict(
        model_identity=extractor.model_identity, warmup_s=warmup, measurements=results,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_revision=assets['dataset_revision'], production_ready=False,
    ), indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
