import ast
from pathlib import Path

def test_lposs_uint8_conversion_contains_accidental_tensor_call():
    """Pins the confirmed legacy defect without importing LPOSS dependencies."""
    tree=ast.parse(Path('segmentation/evaluation/lposs_eval.py').read_text())
    forward=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='forward' and n.lineno>190)
    comprehensions=[n for n in ast.walk(forward) if isinstance(n,ast.ListComp)]
    assert any(isinstance(comp.elt,ast.IfExp) and isinstance(comp.elt.orelse,ast.Call) for comp in comprehensions)
