#!/bin/sh
# Launch the demo without touching your shell configuration.
#
#   sh demo/run.sh            # web UI on http://127.0.0.1:7860
#   sh demo/run.sh cli        # command-line demo on the sample images
#
# It uses the libraries in p2/.pylibs (transformers, gradio, scikit-image) and the
# DINOv2 weights already downloaded to /tmp/hf, so nothing needs installing.

set -e
cd "$(dirname "$0")/.."                 # -> p2/

PY=python3
command -v "$PY" >/dev/null 2>&1 || PY=/usr/local/bin/python3

export PYTHONPATH=".pylibs${PYTHONPATH:+:$PYTHONPATH}"
if [ -d /tmp/hf ]; then export HF_HOME=/tmp/hf; fi
export GRADIO_ANALYTICS_ENABLED=False
# Some environments export proxy / no_proxy values that break httpx inside gradio
# (e.g. a bracketed IPv6 entry such as [::1] is parsed as a malformed port).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY \
      no_proxy NO_PROXY 2>/dev/null || true

echo "python      : $($PY -c 'import sys; print(sys.version.split()[0])')"
echo "PYTHONPATH  : $PYTHONPATH"
echo "HF_HOME     : ${HF_HOME:-<default>}"
$PY -c "import torch, transformers, gradio, cv2; print('deps        : ok (torch', torch.__version__ + ', gradio', gradio.__version__ + ')')" 2>/dev/null || {
  echo
  echo "缺少依赖。请先运行："
  echo "  python3 -m pip install --target .pylibs gradio transformers scikit-image"
  exit 1
}

case "${1:-ui}" in
  cli)
    $PY demo/infer.py --image demo/samples/chewinggum_ng.jpg --category chewinggum
    $PY demo/infer.py --image demo/samples/chewinggum_ok.jpg --category chewinggum
    ;;
  *)
    echo
    echo "打开浏览器访问 http://127.0.0.1:${GRADIO_PORT:-7860}  （Ctrl+C 结束）"
    echo
    exec $PY demo/app.py
    ;;
esac
