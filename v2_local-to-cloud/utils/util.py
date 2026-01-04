import os
from pathlib import Path

def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

        if k and k not in os.environ:
            os.environ[k] = v

            ## ONLY for DEBUG ##
            if "KEY" in k.upper():
                print("LOAD ENV:", k, "*****")
            else:
                print("LOAD ENV:", k, v)