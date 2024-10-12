from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import base64
from silero_TTS.main import generate_silero_speech, get_available_languages, get_language_models, get_model_sample_rates, get_model_example, get_model_speakers
from parler_TTS.main import generate_parler_speech

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        return f.read()

@app.get("/languages")
async def languages():
    return {"languages": get_available_languages()}

@app.get("/models/{language}")
async def models(language: str):
    return {"models": get_language_models(language)}

@app.get("/sample_rates/{language}/{model_id}")
async def sample_rates(language: str, model_id: str):
    return {"sample_rates": get_model_sample_rates(language, model_id)}

@app.get("/example/{language}/{model_id}")
async def example(language: str, model_id: str):
    return {"example": get_model_example(language, model_id)}

@app.get("/speakers/{language}/{model_id}")
async def speakers(language: str, model_id: str):
    return {"speakers": get_model_speakers(language, model_id)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        text = data['text']
        tts_type = data['tts_type']
        
        if tts_type == 'silero':
            language = data['language']
            model_id = data['model_id']
            sample_rate = int(data['sample_rate'])
            speaker = data['speaker']
            audio_data = generate_silero_speech(text, language, model_id, sample_rate, speaker)
        elif tts_type == 'parler':
            description = data['description']
            audio_data = generate_parler_speech(text, description)
        else:
            await websocket.send_json({'event': 'error', 'message': 'Invalid TTS type'})
            continue

        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send_json({'event': 'audio_generated', 'audio': audio_base64})

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=443)
