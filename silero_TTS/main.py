import torch
import io
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {}

def load_model(language, model_id):
    if (language, model_id) not in models:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=model_id)
        model.to(device)
        models[(language, model_id)] = model
    return models[(language, model_id)]

def generate_silero_speech(text, language, model_id, sample_rate, speaker):
    model = load_model(language, model_id)
    
    audio = model.apply_tts(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate)
    
    output = io.BytesIO()
    sf.write(output, audio.cpu().numpy(), sample_rate, format='wav')
    return output.getvalue()

def get_available_languages():
    return ['ru', 'en', 'de', 'es', 'ua']

def get_language_models(language):
    models = {
        'ru': ['v4_ru', 'v3_1_ru', 'ru_v3'],
        'en': ['v3_en', 'v3_en_indic'],
        'de': ['v3_de'],
        'es': ['v3_es'],
        'ua': ['v4_ua', 'v3_ua']
    }
    return models.get(language, [])

def get_model_sample_rates(language, model_id):
    return [8000, 24000, 48000]

def get_model_example(language, model_id):
    examples = {
        'ru': 'В н+едрах т+ундры в+ыдры в г+етрах т+ырят в в+ёдра +ядра к+едров.',
        'en': 'Can you can a canned can into an un-canned can like a canner can can a canned can into an un-canned can?',
        'de': 'Fischers Fritze fischt frische Fische, Frische Fische fischt Fischers Fritze.',
        'es': 'Hoy ya es ayer y ayer ya es hoy, ya llegó el día, y hoy es hoy.',
        'ua': 'К+отики - пухн+асті жив+отики.'
    }
    return examples.get(language, '')

def get_model_speakers(language, model_id):
    speakers = {
        'ru': ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
        'en': ['en_0', 'en_1', 'en_2', 'en_3', 'en_4', 'random'],
        'de': ['eva_k', 'karlsson', 'random'],
        'es': ['es_0', 'es_1', 'es_2', 'random'],
        'ua': ['mykyta', 'random']
    }
    return speakers.get(language, [])
