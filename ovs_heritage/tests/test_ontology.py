import json
import numpy as np
import pytest
from ovs_heritage.ontology import load_ontology, ontology_from_mapping, OntologyError, validate_mask_ids

def test_exact_ontology_and_groups():
    o=load_ontology(); assert [c.id for c in o.classes]==list(range(12)); assert o.ignore_index==255 and 255 not in o.valid_ids
    assert o.by_name('background').id==0; assert o.by_name('advertisements').id==11
    assert 'advertisements' in o.groups['HUMAN_ACTIVITY']; assert 'advertisements' not in o.groups['DAMAGE_MACRO']
    assert len(o.palette)==len(set(o.palette))==12

def test_hash_independent_of_mapping_key_order():
    p='ovs_heritage/configs/heritage_vocab.yaml'; data=json.load(open(p)); reversed_data={k:data[k] for k in reversed(data)}
    assert load_ontology().hash==ontology_from_mapping(reversed_data).hash

def test_invalid_duplicate_id_name_alias():
    data=json.load(open('ovs_heritage/configs/heritage_vocab.yaml'))
    for mutate in ('id','name','alias'):
        x=json.loads(json.dumps(data))
        if mutate=='id': x['classes'][1]['id']=0
        elif mutate=='name': x['classes'][1]['name']='background'
        else: x['classes'][1]['aliases']=['rust']
        with pytest.raises(OntologyError): ontology_from_mapping(x)

def test_unknown_ids_are_explicit_and_11_preserved():
    o=load_ontology(); assert validate_mask_ids(np.array([11,255]),o)=={11,255}
    with pytest.raises(OntologyError,match='17'): validate_mask_ids(np.array([17]),o,'mock.png')
