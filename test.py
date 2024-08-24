import requests

url = "http://127.0.0.1:8000/synthesize/"
payload = {"text": "Hello, world!"}
response = requests.post(url, json=payload)

if response.status_code == 200:
    audio = response.json()["audio"]
    with open("output_audio.wav", "wb") as f:
        f.write(audio)
    print("Audio saved successfully!")
else:
    print(f"Error: {response.json()['detail']}")
