import copy
import screen
from eidolon_sdk.biz.participation import Proposal


def test_annotations_do_not_enter_inference():
 for r in screen.read(screen.HERE/'cases.jsonl'):
  changed=copy.deepcopy(r)
  for key in ('labels','meta','id'):changed[key]='PRIVATE_POISON'
  for reverse in (False,True):assert screen.public(r,reverse)==screen.public(changed,reverse)


def test_every_choice_maps_to_valid_proposal_or_abstention():
 for r in screen.read(screen.HERE/'cases.jsonl'):
  for key in r['questions']['move']['criteria']:
   result=screen.mapped(r['state'],key)
   if result['status']=='abstained':assert result['proposal'] is None
   else:
    proposal=Proposal.model_validate(result['proposal'])
    assert proposal.action in r['state']['allowed_actions']
    assert set(proposal.participants)<=set(r['state']['candidates'])
    if proposal.action=='clarify':assert proposal.instruction.strip()


def test_unknown_and_disallowed_speaker_rejected():
 state={'allowed_actions':['respond','wait','finish'],'candidates':{'M0':{}}}
 for key in ('respond:outsider','clarify:M0'):
  try:screen.mapped(state,key)
  except AssertionError:pass
  else:raise AssertionError('illegal proposal accepted')
