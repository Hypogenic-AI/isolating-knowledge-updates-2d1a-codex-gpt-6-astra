"""Download the exact experiment snapshot; all model files remain in workspace."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
assert Path.cwd()==ROOT,'Run from repository root'
spec=json.loads(Path('results/model_spec.json').read_text())
snapshot_download(spec['id'],revision=spec['revision'],local_dir='artifacts/base_model',allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
