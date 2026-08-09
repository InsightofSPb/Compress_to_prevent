import torch, pytest
from ovs_heritage.ontology import load_ontology
from ovs_heritage.vocabulary import RuntimeClass, build_prototypes, heritage_runtime_vocabulary

def encoder(prompts): return torch.tensor([[len(p), sum(map(ord,p))%19+1, 1.] for p in prompts])
def test_prompt_ensemble_aliases_one_channel_and_order():
    v=(RuntimeClass('mixed',('first','second'),('alias one','alias two')),RuntimeClass('new',('third',)))
    result=build_prototypes(v,encoder,include_alias_prompts=True)
    assert result.prototypes.shape==(2,3); assert result.channel_names==('mixed','new')
    assert torch.allclose(result.prototypes.norm(dim=1),torch.ones(2))
def test_heritage_mixed_unseen_and_arbitrary_order():
    o=load_ontology(); mixed=heritage_runtime_vocabulary(o,['advertisements','crack'])+(RuntimeClass('unseen',('an unseen thing',)),)
    assert build_prototypes(mixed,encoder).channel_names==('advertisements','crack','unseen')
    assert build_prototypes((RuntimeClass('only_new',('new',)),),encoder).prototypes.shape[0]==1
def test_runtime_validation():
    with pytest.raises(ValueError,match='duplicate'): build_prototypes((RuntimeClass('x',('a',)),RuntimeClass('x',('b',))),encoder)
    with pytest.raises(ValueError,match='no prompts'): build_prototypes((RuntimeClass('x',()),),encoder)
