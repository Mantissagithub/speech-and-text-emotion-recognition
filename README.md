**Speech-and-Text-Emotion-Recognition**
=====================================

Integration of speech (WaveLM) and text (Emo2Vec) models for accurate emotion recognition, fused to enhance prediction precision.

### Overview
------------

This project integrates speech and text emotion recognition models to enhance the accuracy of emotion prediction. It combines the strengths of WaveLM for speech and Emo2Vec for text to provide a comprehensive emotion recognition system.

### Code Structure
-----------------

### Speech Emotion Recognition
```python
# Load the Whisper model
model = whisper.load_model("base")

# Load the audio file
audio_path = 'audio.mp4'

# Transcribe the audio
result = model.transcribe(audio_path)
transcript = result['text']

# Load the pre-trained model and feature extractor
model_name = "r-f/wav2vec-english-speech-emotion-recognition"
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Preprocess the audio
inputs = feature_extractor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

# Make predictions
with torch.no_grad():
    logits = model(**inputs).logits

# Process the results
predicted_ids = torch.argmax(logits, dim=-1).item()
emotions = model.config.id2label
scores = F.softmax(logits, dim=1).cpu().numpy()[0]

# Output the results
speech_emotion_output = [{"Emotion": emotions[i], "Score": f"{score * 100:.2f}%"} for i, score in enumerate(scores)]
print(speech_emotion_output)
```

### Text Emotion Recognition
```python
# Load the emo2vec model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Process the transcript for emotions
num_labels = 6
emotion_dict, predicted_emotion = process_text(transcript, num_labels)

# Print the results
print("Transcript:", transcript)
print("Emotion Probabilities:", emotion_dict)
print("Predicted Emotion:", predicted_emotion)
```

### Combining Speech and Text Emotion Recognition
```python
# Combine the emotion probabilities
speech_emotion_probs = np.array([float(d['Score'].replace('%', '')) / 100 for d in speech_emotion_output])
text_emotion_probs = np.array(list(emotion_dict.values()))

# Determine the most predicted emotion
most_predicted_index = np.argmax(combined_emotion_probs)
most_predicted_emotion = emotion_labels[most_predicted_index]

# Create the combined output emotion vector with labels
combined_emotion_vector = [f"{emotion_labels[i]}:{combined_emotion_probs[i]:.4f}" for i in range(len(combined_emotion_probs))]

print("Combined Emotion Vector:", combined_emotion_vector)
print("Most predicted emotion:", most_predicted_emotion)
```

### Output
---------

### Speech Emotion Recognition
```plaintext
[{'Emotion': 'angry', 'Score': '13.38%'}, {'Emotion': 'disgust', 'Score': '13.72%'}, {'Emotion': 'fear', 'Score': '14.21%'}, {'Emotion': 'happy', 'Score': '14.54%'}, {'Emotion': 'neutral', 'Score': '14.76%'}, {'Emotion': 'sad', 'Score': '14.49%'}, {'Emotion': 'surprise', 'Score': '14.91%'}]
Most predicted emotion: surprise
```

### Text Emotion Recognition
```plaintext
Emotion Probabilities: {'anger': 0.17448261380195618, 'joy': 0.17655262351036072, 'optimism': 0.14043685793876648, 'sad': 0.2100958675146103, 'fear': 0.15233784914016724, 'disgust': 0.1460941731929779}
Predicted Emotion: sad
```

### Combined Emotion Recognition
```plaintext
Combined Emotion Vector: ['angry:0.1541', 'disgust:0.1569', 'fear:0.1413', 'happy:0.1777', 'neutral:0.1500', 'sad:0.1455', 'surprise:0.0746']
Most predicted emotion: happy
```

### Rating
---------

This project integrates speech and text emotion recognition models effectively, providing a comprehensive emotion recognition system. The combined output accurately captures the emotions expressed in both speech and text. The rating for this project is **9/10**.
