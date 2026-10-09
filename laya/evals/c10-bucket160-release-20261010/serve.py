"""Isolated HTTP candidate; original production artifact remains untouched."""
from pathlib import Path
from aiohttp import web
from eidolon_models_laya.config import Settings
from eidolon_models_laya.engine import load_engine
from eidolon_models_laya.service import create_app
s=Settings(service='smart_home', backend='rknn',
    model_dir=Path('/tmp/laya-efficiency-20261009/model160'),
    host='127.0.0.1',port=18774,max_len=512,head_max_len=512,max_pending=1,
    rknn_placement='1:128,160,256,384,512|2:128,160,256')
e,m=load_engine(s)
web.run_app(create_app(e,s,{'model':m.name,'repo_id':m.repo_id,'revision':m.revision,'subfolder':m.subfolder}),host=s.host,port=s.port)
