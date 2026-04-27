#!/bin/bash
cd /root/.openclaw/workspace/clawlite
exec gunicorn -c gunicorn_config.py "proxy:app"
