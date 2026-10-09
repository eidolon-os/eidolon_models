"""Runs an interpreter only; validates the production replay sink without a command executor."""
import asyncio
import json
import stat
from pathlib import Path
from eidolon_agent.config import load_settings
from eidolon_agent.domain.interpretation import InterpretationConfig,InterpretationService
from eidolon_agent.infra.interpretation import JsonlInterpretationRecorder,LayaInterpreter
from eidolon_sdk.biz.interpretation import InterpretationRequest,Candidate

async def main():
    settings=load_settings(yaml_path=Path('/etc/eidolon/agent.yaml'))
    recorder=JsonlInterpretationRecorder(settings.smarthome.interpretation_record_path)
    model=LayaInterpreter(settings.smarthome.laya.url)
    service=InterpretationService({'laya':model},InterpretationConfig(primary='laya'),recorder=recorder)
    req=InterpretationRequest(interpretation_id='deployment-smoke-20261010',domain='smarthome',
        utterance='这是发布诊断文本，不要执行设备操作。',
        candidates=(Candidate(ref='test.tv',name='测试电视',kind='media'),),timeout_ms=1000)
    try:
        result=await service.interpret(req)
        await service.drain()
        row=json.loads(recorder.path.read_text().splitlines()[-1])
        assert row['request']['interpretation_id']==req.interpretation_id
        assert row['primary']['result']['model_version'].endswith('@e8254243')
        assert stat.S_IMODE(recorder.path.stat().st_mode)==0o600
        assert 'answers_json' in row['primary']['result']['diagnostics']
        print(json.dumps({'path':str(recorder.path),'mode':'0600','model_version':result.model_version,
            'recorded':True,'executor_called':False},ensure_ascii=False))
    finally:await model.aclose()

asyncio.run(main())
