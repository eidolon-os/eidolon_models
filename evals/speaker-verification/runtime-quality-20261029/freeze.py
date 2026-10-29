"""Real inference + recognition + signed turn freeze, with a fixed test directory."""
import argparse
import asyncio
import hashlib
import json
import time
import wave
from pathlib import Path

from eidolon.livekit.agent.session.conversation_speaker import ConversationSpeaker
from eidolon.livekit.agent.session.speaker_recognition import SpeakerRecognizer
from eidolon.livekit.common.audio_evidence import AudioEvidence
from eidolon.livekit.plugins.speaker_verification.inference import SpeakerInferenceEngine
from eidolon.livekit.plugins.speaker_verification.reference_extractor import ReferenceSpeakerExtractor
from eidolon.livekit.plugins.speaker_verification.scoring import compare_speakers
from eidolon_sdk.biz.speaker_attestation import verify_observation
from eidolon_sdk.biz.speaker_identity import SpeakerCandidates, SpeakerRecognitionPolicy
from eidolon_sdk.biz.voiceprint import SpeakerComparisonRequest


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = json.loads(args.assets.read_text())
    row = next(x for x in assets['samples'] if x['speaker'] == 'S0003')
    path = args.assets.parent / row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        pcm = wav.readframes(wav.getnframes())
    engine = SpeakerInferenceEngine(ReferenceSpeakerExtractor(args.model))
    results = []
    try:
        await engine.start()
        sample = await engine.extract(pcm)
        snapshot = SpeakerCandidates(owner_id='test-owner', revision='b'*64, persons=[dict(
            person_id='test-person', person_revision=1, display_name='test', profile={},
            templates=[dict(profile_id='test-template', revision=1, sample=sample)])])
        policy = SpeakerRecognitionPolicy(version='offline-self-match-only', model_id=sample.model_id,
            weights_hash=sample.weights_hash, preprocessing_version=sample.preprocessing_version,
            threshold=.8, min_person_margin=.1, min_speech_ms=1500)

        class Directory:
            async def candidates(self, owner_id):
                assert owner_id == 'test-owner'
                return snapshot

            async def extract(self, data, *, min_speech_ms):
                return await engine.extract(data, min_speech_ms)

            async def compare(self, probe, templates):
                return compare_speakers(SpeakerComparisonRequest(probe=probe, templates=templates))

        class Sink:
            def __init__(self):
                self.tokens = []

            def set_speaker_observation(self, token):
                self.tokens.append(token)

        secret = 'isolated-real-inference-freeze-key'
        for number in range(3):
            turn = f'turn-{number}'
            feedback, sink = [], Sink()
            observer = ConversationSpeaker(recognizer=SpeakerRecognizer(Directory(), policy),
                owner_id='test-owner', session_id='test-session', secret=secret, publish=feedback.append)
            try:
                evidence = AudioEvidence(source='public-audio', generation=1, candidate=turn,
                    sample_rate=16000, start=0, end=len(pcm)//2, pcm=pcm,
                    complete=True, reason=None, session_id='test-session')
                observer.begin_evidence(evidence, turn)
                task = observer._task
                await asyncio.sleep(0)  # Let real extraction enter its executor.
                assert not task.done()
                started = time.perf_counter()
                frozen = await observer.result(turn)
                observer.deliver(frozen, text='offline test', llm=sink, timeline=None)
                freeze_ms = (time.perf_counter()-started)*1000
                assert frozen.person_id is None and frozen.reason == 'recognition_pending'
                await task
                await asyncio.sleep(0)
                late = feedback[-1]
                assert late.status == 'identified' and late.person_id == 'test-person'
                assert await observer.result(turn) == frozen
                observer.deliver(frozen, text='offline test', llm=sink, timeline=None)
                assert len(sink.tokens) == 1
                signed = verify_observation(sink.tokens[0], secret=secret, text='offline test',
                    owner_id='test-owner', session_id='test-session', turn_id=turn)
                assert signed.person_id is None
                assert (await observer.result('unobserved-next-turn')).person_id is None
                results.append(dict(freeze_ms=freeze_ms,
                    completion_after_freeze_ms=(time.perf_counter()-started)*1000,
                    delivered_status=signed.status, late_status=late.status,
                    signed_deliveries=len(sink.tokens)))
            finally:
                await observer.aclose()
    finally:
        await engine.close()
    output = dict(measurements=results, sample_sha256=row['sha256'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Self-match and fixed in-memory directory; no Data HTTP, gRPC, LLM or Memory. Not accuracy calibration.')
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(results))


if __name__ == '__main__':
    asyncio.run(main())
