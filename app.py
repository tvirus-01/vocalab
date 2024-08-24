from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import xttsv2
from waveglow.denoiser import Denoiser
import soundfile as sf
import librosa
import numpy as np

# Define the request model
class TTSRequest(BaseModel):
    text: str

# Initialize the FastAPI app with the name "Vocalab"
app = FastAPI(title="Vocalab", description="An API for real-time Text-to-Speech using XTTSv2 and WaveGlow.", version="1.0")

# Load pre-trained XTTSv2 model
model = xttsv2.load_model('finetuned_xttsv2_model.pth')
model.eval()  # Set the model to evaluation mode

# Load WaveGlow model and denoiser
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()
denoiser = Denoiser(waveglow)

# Define the synthesize function
def synthesize_audio(mel_spectrogram, waveglow, denoiser, sigma=0.666):
    with torch.no_grad():
        mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)  # Add batch dimension
        audio = waveglow.infer(mel_spectrogram, sigma=sigma)
        audio = denoiser(audio, strength=0.01)[:, 0]  # Apply denoiser
        return audio.cpu().numpy().flatten()

@app.post("/synthesize/")
def synthesize(request: TTSRequest):
    text = request.text

    try:
        # Generate mel-spectrogram with input text
        mel_spectrogram = model(text)  # Assuming the model takes text directly
        
        # Generate audio from mel-spectrogram
        synth_audio = synthesize_audio(mel_spectrogram[0].detach().cpu().numpy(), waveglow, denoiser)
        
        # Save the audio temporarily
        temp_audio_path = 'output_synth_audio.wav'
        sf.write(temp_audio_path, synth_audio, samplerate=22050)  # Adjust the sampling rate as necessary
        
        # Load the audio file to return as bytes
        audio, sr = librosa.load(temp_audio_path, sr=None)
        audio_bytes = np.array(audio, dtype=np.float32).tobytes()
        
        return {"audio": audio_bytes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
