import importlib.util
import sys
from pathlib import Path

def load_profile(profile_name):
    # profilesフォルダから .py ファイルを探してロードする
    base_path = Path(__file__).parent
    target_path = base_path / f"{profile_name}.py"
    
    if not target_path.exists():
        raise FileNotFoundError(f"Profile not found: {target_path}")
    
    spec = importlib.util.spec_from_file_location(profile_name, target_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[profile_name] = module
    spec.loader.exec_module(module)
    
    return module.PROFILE