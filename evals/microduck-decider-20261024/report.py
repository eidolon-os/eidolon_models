import json,pathlib,collections,statistics
P=pathlib.Path(__file__).resolve().parent
s=json.loads((P/'summary.json').read_text());rows=[json.loads(x) for x in (P/'results.jsonl').read_text().splitlines()];f=json.loads((P/'fixtures.json').read_text());cases={c['id']:c for c in f['cases']};env=json.loads((P/'environment.json').read_text())
pct=lambda x:f'{100*x:.1f}%'
out=['# Microduck 语音文本决策基线 · 2026-10-24','', '## 范围与方法','',f'36 条人工编写的合成中文案例，每模型重复 3 次；共 {len(rows)} 个计分请求。另有每模型 1 次预热，不计入延迟。标签、能力设定、问题、0.8 阈值在调用前固定，见 fixtures.json 与 SHA256。重复不视为新增独立样本。本轮无训练、无 ASR、无机器人调用。','', '这是封闭候选分类 + 确定性编译的基线，不是任意 JSON 生成测试。五题同时询问 route/action/target/distance/speed。所有模型使用相同中文 state、问题和候选。全局任务规则写入 state；不代表各模型最优提示格式。所有调用单请求串行，案例和模型顺序按固定种子打乱。客户端 urllib 不复用持久连接，因此时间包括连接建立、网络与服务推理；公网与本机不是同硬件速度比较。','',f'本机：{env["cpu"]}，内存 {int(env["memory_bytes"])/2**30:.0f} GiB，{env["platform"]}。Laya：{env["laya"]}。D3/pplx 使用用户文档指定在线服务，未取得云端 checkpoint hash。','', '模拟能力：两台在线机器人，支持距离闭环，速度0.05–0.3米/秒、距离不超过2米；这不是原厂能力声明。没有唯一目标或距离缺失时询问，复合任务转规划器且不发部分动作；非命令字段按固定规则归一。','', '## 结果','', '| 模型 | API错误 | 五字段全对 | route准确率 | P50/P95 ms | 无阈值：正确执行/应执行 | 无阈值：错发/发出 | 0.8门控：正确执行/应执行 | 0.8门控：错发/发出 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for m,v in s.items():
 out.append(f'| {m} | {v["errors"]}/{v["requests"]} | {pct(v["exact"])} | {pct(v["fields"]["route"])} | {v["p50_ms"]:.0f}/{v["p95_ms"]:.0f} | {pct(v["raw"]["execution_recall"])} | {v["raw"]["wrong_emitted"]}/{v["raw"]["emitted"]} | {pct(v["gated"]["execution_recall"])} | {v["gated"]["wrong_emitted"]}/{v["gated"]["emitted"]} |')
out+=['','五字段全对是严格格式指标：无动作时仍输出目标名也计错，即使不造成错发。命令指标按完整 device/op/args 判定；该执行的命令被门控掉计漏执行。0.8门控要求五个所选选项的概率全部≥0.8，未经本场景校准，可能因无关槽位降低覆盖。P95采用排序数组下标 floor(0.95*(N-1))。','', '## 分字段','', '| 模型 | route | action | target | distance | speed |','|---|---:|---:|---:|---:|---:|']
for m,v in s.items():out.append('| '+m+' | '+' | '.join(pct(v['fields'][k]) for k in f['questions'])+' |')
out+=['','## 首轮逐例结果','', '| ID | 当前话语 | 期望route/action | Laya | D3 | pplx |','|---|---|---|---|---|---|']
for cid,c in cases.items():
 cells=[]
 for m in ('laya','d3','pplx'):
  r=next(r for r in rows if r['id']==cid and r['model']==m and r['repeat']==0)
  cells.append('ERROR' if 'error'in r else ('✓' if r['exact'] else '✗')+' '+r['pred']['route']+'/'+r['pred']['action'])
 out.append('| '+cid+' | '+c['text']+' | '+c['expected']['route']+'/'+c['expected']['action']+' | '+' | '.join(cells)+' |')
out+=['','## 命令输出与错误实例','']
for m in ('laya','d3','pplx'):
 out+=['### '+m,'']
 bad=[r for r in rows if r['model']==m and r['repeat']==0 and r.get('unsafe_raw')]
 out+=['无阈值错发案例：'+(', '.join(r['id'] for r in bad) or '无')+'。','']
 for cid in ('C03','C16','C19','C30','C34'):
  r=next(r for r in rows if r['id']==cid and r['model']==m and r['repeat']==0)
  out+=['```json',json.dumps({'case':cid,'text':cases[cid]['text'],'expected_command':r.get('expected_command'),'pred':r.get('pred'),'prob':r.get('prob'),'command_raw':r.get('command_raw'),'command_gated':r.get('command_gated')},ensure_ascii=False,indent=2),'```','']
out+=['## 局限与下一步','', '- C34 的0.37米不在距离候选里：即使模型选对 other，编译器仍不能生成该数值命令。这是架构表达能力缺口；应采用独立数值/单位抽取或受约束结构化解析，不能无限枚举参数。','- 全局规则、设备能力和用户话语放在同一 state，可能污染小模型判断；本轮未按结果反复调提示。后续应在独立开发集比较紧凑输入、按题分配上下文与问题内规则，再用新冻结测试集验收。','- 0.8来自固定基线设定，并非已验证的安全阈值。尤其停止命令要独立评估漏检、误停和响应延迟，不能直接套普通动作门槛。','- 36条样本不足以估计线上准确率；三次重复主要用于延迟和稳定性。需新增真实录音ASR转写、口语修正、方言、打断、设备离线和任务状态竞争。','- 推荐先复用Eidolon Agent工具执行路径，把决策结果视为提案；能力、范围、状态、幂等、TTL和反馈仍由确定性执行层检查。','- 文件：benchmark.py 为可复跑脚本；fixtures.json 为完整标签与输入；results.jsonl 为每次完整响应/概率/延迟/命令；summary.json 为统计；warmup.json 为预热；environment.json 为环境。','']
(P/'REPORT.md').write_text('\n'.join(out))
print(P/'REPORT.md')
