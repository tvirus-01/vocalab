# Vocalab

Vocalab is a real-time Text-to-Speech (TTS) API built using FastAPI, XTTSv2, and WaveGlow. This service allows you to convert text input into high-quality speech audio.

## Features

- Real-time Text-to-Speech (TTS) conversion
- API built with FastAPI
- Uses XTTSv2 for text to mel-spectrogram conversion
- WaveGlow for mel-spectrogram to audio synthesis
- Denoiser to improve audio quality

## Installation

### Prerequisites

- Python 3.7+
- **pip**: Python package installer

### Dependencies

Install the required Python packages:

```sh
pip install fastapi uvicorn torch torchvision torchaudio pydantic soundfile librosa numpy
```

## Usage

#### Running the Server

1. Clone the repository:
```sh
git clone https://github.com/yourusername/vocalab.git
cd vocalab
```
2. Save the provided script as app.py.

3. Run the FastAPI server using uvicorn:
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

Your API will be available at http://127.0.0.1:8000.

## API Endpoints

#### POST /synthesize/
This endpoint accepts a JSON payload containing the text to be synthesized and returns the synthesized audio in bytes.

- Request Body:
```json
{
  "text": "Hello, world!"
}
```

- Response:
```json
{
  "audio": "AUDIO_BYTES"
}
```

## Example Request
Use ```test.py``` to make a request to the API using Python

## Contributing
Contributions are welcome! Please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.