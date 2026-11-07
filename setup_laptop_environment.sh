
#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$PROJECT_ROOT"

python3 -m venv \
--system-site-packages \
.venv-stt

PYTHON="$PROJECT_ROOT/.venv-stt/bin/python"

"$PYTHON" -m pip install \
-r requirements-laptop.txt

"$PYTHON" -m pip install \
--no-deps \
openwakeword==0.6.0

"$PYTHON" -m pip install \
--force-reinstall \
--no-deps \
numpy==1.26.4

"$PYTHON" -u -c '
from openwakeword.utils import download_models

print("Downloading openWakeWord resources...")
download_models()
print("openWakeWord resources ready")
'

echo "Laptop environment ready"
