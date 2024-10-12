import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import numpy as np
import io
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

def generate_parler_speech(prompt, description):
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Normalize audio to float32 range (-1, 1)
    audio_arr = audio_arr / np.max(np.abs(audio_arr))
    
    # Convert to WAV format
    output = io.BytesIO()
    sf.write(output, audio_arr, model.config.sampling_rate, format='wav')
    return output.getvalue()

if __name__ == "__main__":
    prompt = "Hey, how are you doing today?"
    description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

    audio_data = generate_parler_speech(prompt, description)
    print(f"Generated audio data of length: {len(audio_data)} bytes")
