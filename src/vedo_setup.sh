#!/bin/bash
#
set -x
export DISPLAY=:99.0
Xvfb :99 -screen 0 1024x1024x24 > /dev/null 2&>1 &
sleep 3
set +x
exec "$@"
