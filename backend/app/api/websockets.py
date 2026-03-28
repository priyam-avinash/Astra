from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
import os
import redis.asyncio as aioredis # type: ignore

router = APIRouter()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

@router.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    await websocket.accept()
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("market_ticks")
    
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                try:
                    data = json.loads(message["data"])
                    await websocket.send_json(data)
                except Exception:
                    pass
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        await pubsub.unsubscribe("market_ticks")
        await redis_client.aclose()
    except Exception as e:
        print(f"WebSocket Error: {e}")
        try:
            await pubsub.unsubscribe("market_ticks")
            await redis_client.aclose()
        except:
            pass
