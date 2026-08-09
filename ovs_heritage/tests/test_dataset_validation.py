import csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from ovs_heritage.ontology import load_ontology, ontology_from_mapping
from ovs_heritage.validate_dataset import validate_splits, main

def save(path, values): Image.fromarray(np.array(values,dtype=np.uint8)).save(path)
def test_report_unknown_filename_and_json(tmp_path):
    good=tmp_path/'good.png'; bad=tmp_path/'bad.png'; save(good,[[0,1,10,11,255]]); save(bad,[[42]])
    report=validate_splits({'train':tmp_path},load_ontology()); assert not report['valid']; assert str(bad) in str(report); assert 11 in report['splits']['train']['unique_ids']
    out=tmp_path/'report.json'; assert main(['--train',str(tmp_path),'--output',str(out),'--strict'])==1
    assert json.loads(out.read_text())['splits']['train']['mask_count']==2

def test_v1_rejects_11_and_split_facade_overlap(tmp_path):
    mask=tmp_path/'mask.png'; save(mask,[[11]])
    raw=json.load(open('ovs_heritage/configs/heritage_vocab.yaml')); raw['version']='heritage_facades_v1_11classes'; raw['classes']=raw['classes'][:11]
    for g in raw['groups'].values():
        if 'advertisements' in g:g.remove('advertisements')
    v1=ontology_from_mapping(raw); assert not validate_splits({'train':tmp_path},v1)['valid']
    manifests=[]
    for split in ('train','test'):
        p=tmp_path/f'{split}.csv'
        with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['mask_path','facade_id']); w.writeheader(); w.writerow({'mask_path':'mask.png','facade_id':'same'})
        manifests.append(p)
    assert any('overlap' in e for e in validate_splits({'train':manifests[0],'test':manifests[1]},load_ontology())['errors'])

def test_absent_advertisements_is_warning(tmp_path):
    save(tmp_path/'no_ads.png',[[0,1,255]]); report=validate_splits({'val':tmp_path},load_ontology())
    assert report['valid']; assert any('ADVERTISEMENTS' in w for w in report['warnings'])
