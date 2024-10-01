import cohere
import speech_recognition as sr
import pyttsx3
import time

# Initialize the Cohere client with your API key
cohere_api_key = "5VxilEqsCBq9ZfpeYOP5lN37r9t5F3z3KoP26u0h"  # Replace with your API key
co = cohere.Client(cohere_api_key)

# Initialize the speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# List available voices
voices = engine.getProperty('voices')

# Print available voices and choose a higher-pitched one
for index, voice in enumerate(voices):
    print(f"Voice {index}: {voice.name}")

# Set the voice to a higher-pitched one (adjust the index based on your system)
baby_voice_index = 1  # Change this index based on the output above
engine.setProperty('voice', voices[baby_voice_index].id)

# Set speech rate to a higher value for a more playful sound
speech_rate = 180  # Adjust as needed for faster speech
engine.setProperty('rate', speech_rate)

def speak(text):
    print(f"Chatbot: {text}")  # Print the response
    engine.say(text)
    engine.runAndWait()

def chatbot_response(prompt):
    try:
        response = co.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            stop_sequences=["\n"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        try:
            audio = recognizer.listen(source, timeout=5)  # Timeout after 5 seconds
            print("Finished listening, processing audio...")
            return audio
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start.")
            return None
        except Exception as e:
            print(f"Error during listening: {e}")
            return None

# Simple Chatbot Loop
introduction = "My name is Sam! I am a brilliant, sophisticated, AI-assistant chatbot trained to assist human users by providing thorough responses. I am powered by Command, a large language model built by the company Cohere. Is there anything I can help you with today?"

print("Chatbot: Hello! Please say 'hello sam' to start a conversation.")
speak("Hello! Please say 'hello sam' to start a conversation.")  # Speak the greeting
activated = False  # Flag to check if chatbot is activated

while True:
    audio = listen()
    if audio is not None:
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You: {user_input}")
            speak(f"You said: {user_input}")  # Speak the user's input
            
            # Check if the activation phrase is said
            if "hello sam" in user_input.lower():
                activated = True
                speak(introduction)  # Introduce the chatbot
                print(introduction)  # Print the introduction
                speak("Chatbot activated. How can I assist you?")
                print("Chatbot activated. How can I assist you?")
            elif activated:
                # Generate response from the chatbot only if activated
                response = chatbot_response(user_input)
                speak(response)
                
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            speak("Sorry, I could not understand the audio.")  # Speak error
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            speak(f"Could not request results; {e}")  # Speak error
    time.sleep(1)  # Optional: wait before the next listening cycle
