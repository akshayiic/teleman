#!/bin/bash
# Starts the AVTR-1 renderer (:8000) and the teleman bridge (:8770) on the GPU machine;
# Ctrl-C stops both. Expects avtr1_bridge.py copied into the avtr-1 checkout.
cd "${AVTR1_DIR:-$HOME/avtr-1}"
PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
export AVTR1_BG="${AVTR1_BG:-minimal_office}"  # backdrop behind cut-out avatars (see /avatars for the list)
trap 'kill 0' INT TERM EXIT
$PIXI run -e renderer python -m avtr1_renderer.api.app 2>&1 | sed -u 's/^/[renderer] /' &
echo "Waiting for the renderer to load models..."
until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 2; done
echo "Renderer ready: $(curl -s http://127.0.0.1:8000/avatars)"
$PIXI run -e renderer python avtr1_bridge.py 2>&1 | sed -u 's/^/[bridge] /' &
wait
