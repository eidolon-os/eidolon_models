"""Real configured LLM, synthetic home and in-memory executor. No Hub connection."""
import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from eidolon_agent.config import load_settings
from eidolon_agent.infra.llm.providers import LiteLLMProvider
from eidolon_agent.infra.smarthome.llm_fallback import LlmHomeFallback
from eidolon_agent.domain.smarthome.command import SmartHomeCommand
from eidolon_agent.domain.smarthome.context import HomeContext
from home_fakes import FakeDirectory,FakeExecutor,home_registry,OWNER
from eidolon_sdk.biz.interpretation import InterpretationResult, Proposal, Action
from eidolon_agent.infra.interpretation.adapters.laya import LayaInterpreter

class Capture:
    def __init__(self, inner):
        self.inner=inner
        self.last=None
    async def interpret(self, request):
        result=await self.inner.interpret(request)
        self.last={'request':request.model_dump(mode='json'),'result':result.model_dump(mode='json')}
        return result

class Abstain:
    async def interpret(self,request):
        return InterpretationResult(interpretation_id=request.interpretation_id,status='abstained',
            model_version='test-force-fallback',policy_version='test')

# Sequential synthetic scenario; explicit washer stop avoids labeling the original ambiguous question.
CASES=[
 ('打开音箱。','living.speaker','on_off','on',{}),
 ('调大到音量30%。','living.speaker','volume','set',{'value':30}),
 ('音量降低10。','living.speaker','volume','step',{'delta':-10}),
 ('关了吧。','living.speaker','on_off','off',{}),
 ('停止洗衣机。','balcony.washer','operational','stop',{}),
 ('开一下电视。','living.tv','on_off','on',{}),
 ('关了电视。','living.tv','on_off','off',{}),
 ('电视打开。','living.tv','on_off','on',{}),
 ('声音小一点。','living.tv','volume','step',{'delta':-10}),
 ('电视的声音小一点。','living.tv','volume','step',{'delta':-10}),
 ('主卧空调打开。','master.ac','on_off','on',{}),
 ('关闭空调。','master.ac','on_off','off',{}),
 ('关闭空调。','master.ac','on_off','off',{}),
 ('客厅空调打开。','living.ac','on_off','on',{}),
 ('关闭空调。','living.ac','on_off','off',{}),
 ('打开音箱。','living.speaker','on_off','on',{}),
 ('音量调到40%。','living.speaker','volume','set',{'value':40}),
 ('音量增加10。','living.speaker','volume','step',{'delta':10}),
 ('音量降到20%。','living.speaker','volume','set',{'value':20}),
 ('音量降低5。','living.speaker','volume','step',{'delta':-5}),
]
async def main(args):
    load_dotenv('/etc/eidolon/agent.env',override=False)
    load_dotenv('/etc/eidolon/host.env',override=False)
    settings=load_settings(yaml_path=Path('/etc/eidolon/agent.yaml'))
    m=next(m for m in settings.llm.models if m.name==settings.llm.default_model)
    llm=LiteLLMProvider(model=m.name,api_key=m.resolved_api_key(),api_base=m.api_base,
        timeout_s=m.timeout_s,max_retries=settings.llm.max_retries,thinking=m.thinking)
    model=LayaInterpreter(args.laya_url) if args.laya_url else None
    primary=Capture(model or Abstain())
    results=[]
    cases = CASES if args.suite == 'conversation' else [(text,None,None,None,{}) for text in (
        '不要关闭音箱。','别调高音量。','朋友说打开电视。','明天帮我打开电视。',
        '取消刚才的请求。','如果要关闭电视，应该怎么说？')]
    for repeat in range(3):
        directory=FakeDirectory(home_registry());executor=FakeExecutor(directory)
        command=SmartHomeCommand(directory=directory,executor=executor,interpreter=primary,fallback=LlmHomeFallback(llm))
        context=HomeContext()
        for i,(text,target,trait,action,params) in enumerate(cases):
            if args.suite == 'negative':
                context.clear()
                context.remember('打开音箱', Proposal(intent='control',target_status='resolved',targets=('living.speaker',),action=Action(trait='on_off',command='on')),response='已打开智能音箱',outcome='executed')
            n=len(executor.commands)
            voice=await command.handle(OWNER,None,f'isolated-{repeat}-{i}',text,context=context)
            actual=[list(c) for c in executor.commands[n:]]
            expected=[[target,trait,action,params]] if target else []
            row={'repeat':repeat,'text':text,'expected':expected,'commands':actual,'voice':voice.model_dump(mode='json'),'passed':actual==expected,'primary':primary.last}
            results.append(row)
            args.out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print('repeat',repeat,'passed',sum(r['passed'] for r in results),'total',len(results),flush=True)
    print('FINAL',sum(r['passed'] for r in results),len(results),flush=True)
    if model:await model.aclose()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--suite',choices=['conversation','negative'],default='conversation')
    p.add_argument('--laya-url')
    asyncio.run(main(p.parse_args()))
