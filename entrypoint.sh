#!/bin/bash

# Default to FastAPI if no APP_MODE specified
APP_MODE=${APP_MODE:-fastapi}

if [ "$APP_MODE" = "api" ]; then
    exec uvicorn melo.fastapi_server:app --host "0.0.0.0" --port "8888" --reload
else
    exec python ./melo/app.py --host "0.0.0.0" --port "8888"
fi