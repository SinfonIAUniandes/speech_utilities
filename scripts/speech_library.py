# This file contains the speech library class which is used to have all the functions related to the speech library
import os
import soundfile as sf
import speech_recognition as sr
import numpy as np
import nltk
import ConsoleFormatter
import rospkg
import rospy    
# === Import messages ===
from robot_toolkit_msgs.msg import leds_parameters_msg

# === Imports whisper ===
import time
import torch
import whisper

# === Parameters whisper ===
def setLedsColor(r,g,b):
    """
    Function for setting the colors of the eyes of the robot.
    Args:
    r,g,b numbers
        r for red
        g for green
        b for blue
    """
    ledsPublisher = rospy.Publisher('/leds', leds_parameters_msg, queue_size=10)
    ledsMessage = leds_parameters_msg()
    ledsMessage.name = "FaceLeds"
    ledsMessage.red = r
    ledsMessage.green = g
    ledsMessage.blue = b
    ledsMessage.time = 0
    ledsPublisher.publish(ledsMessage)  #Inicio(aguamarina), Pepper esta ALSpeechRecognitionStatusPublisherlista para escuchar
    rospy.sleep(0.2)

def transcribe(file_path, model):
    """
    Input:
    file_path: path of the .wav file to transcribe
    model: Whisper model instance to transcribe audio
    ---
    Output:
    response of the local model with the transcription of the audio
    ---
    Use the local version of whisper for transcribing short audios
    """
    t1 = float(time.perf_counter() * 1000)
    torch.cuda.empty_cache()
    result = model.transcribe(file_path)
    torch.cuda.empty_cache()
    t2 = float(time.perf_counter() * 1000)
    print("Local [ms]: ", float(t2-t1))
    return result["text"]

def transcribe_cloud(file_path, client):
    """
    Input:
    file_path: path of the .wav file to transcribe
    client: AzureOpenAI's client instance to transcribe the audio
    ---
    Output:
    response of the cloud model with the transcription of the audio
    ---
    Use the cloud version of whisper for transcribing short audios
    """
    audio_file = open(file_path, "rb")
    result = client.audio.transcriptions.create(
        model = "whisper",
        file = audio_file,
    )
    return result.text

def transcribe_spanish(file_path, model):
    """
    Input:
    file_path: path of the .wav file to transcribe
    model: Google model instance to transcribe audio
    ---
    Output:
    response of the cloud model with the transcription of the audio in spanish
    ---
    Use the cloud version of Google Recognizer for transcribing short audios
    """
    with sr.AudioFile(file_path) as source:
            audio = model.record(source)
    transcription = "None"
    try:
        transcription = model.recognize_google(audio,language="es")
    except sr.UnknownValueError:
        print("Google no entendio")
    except sr.RequestError:
        print("Error en la peticion")
    return transcription

def load_model(model_size = "small"):
    PATH_SPEECH_UTILITIES = rospkg.RosPack().get_path('speech_utilities')
    PATH_DATA = PATH_SPEECH_UTILITIES+'/data'
    in_memory = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    print(consoleFormatter.format(f"Transcribing audio in {device.upper()}", "OKBLUE"))
    if not os.path.exists(PATH_DATA+f"/{model_size}.pt"): 
        print("Model not found, downloading it")
        in_memory = False
    model_whisp = whisper.load_model(model_size, in_memory= in_memory, download_root = PATH_DATA)
    return model_whisp, device

def save_recording(buffer,file_name,sample_rate):
    """
    Input:
    buffer: audio buffer
    file_name: name of the file
    sample_rate: sample rate of the audio
    ---
    Saves the audio buffer in the data folder with the name file_name
    """
    PATH_SPEECH_UTILITIES = rospkg.RosPack().get_path('speech_utilities')
    PATH_DATA = PATH_SPEECH_UTILITIES+'/data'
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    print(consoleFormatter.format(f"Saving audio to: {file_name}", "WARNING"))
    # Check if data folder exists, if not, create it
    if not os.path.exists(PATH_DATA): 
        os.makedirs(PATH_DATA) 
    # Convert the audioBuffer into a array 
    audio_buffer = np.array(buffer) 
    audio_buffer = audio_buffer.astype(float)
    audio_buffer = np.asarray(audio_buffer, dtype=np.int16)
    # Save the audio in the folder with a specific name
    path = PATH_DATA+"/"+file_name+".wav"
    sf.write(path,audio_buffer,sample_rate,closefd=True)
        
def gpt(client,messages,temperature):
    """
    Input:
    messages: list of dictionaries
    temperature: temperature of the model
    ---
    Output:
    response of the model
    ---
    This function is used to make a request to the GPT model given a list of dictionaries
    """
    try:
        prediction = client.chat.completions.create(
            model="GPT-4o", 
            messages=messages, 
            temperature=temperature, 
            max_tokens=500
        )
        response = {
            "content": prediction.choices[0].message.content,
            "role": prediction.choices[0].message.role,
        }
    except:
        response = {
            "content": "Disculpa, no puedo responder eso",
            "role": "error",
        }
    return response

def nltk_processing(text):
    """
    Input:
    text: the text to process
    ---
    Output:
    tuple containing two elements:
        - A list of tokens.
        - A list of tuples, where each tuple contains a token and its part-of-speech (POS) tag.
            For example, [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'),
            ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]
            In this example, 'DT' represents a determiner, 'JJ' an adjective, 'NN' a noun, and 'VBZ' a verb in third person singular present.
    ---
    Perform text processing using NLTK
    """
    if text == "":
        return [], []
    if not os.path.exists(nltk.data.find("tokenizers/punkt")):
        nltk.download('punkt')
    if not os.path.exists(nltk.data.find("taggers/averaged_perceptron_tagger")):
        nltk.download('averaged_perceptron_tagger')
    try:
        tokens = nltk.tokenize.word_tokenize(text)
        etiquetas = nltk.pos_tag(tokens)
        return tokens, etiquetas
    except Exception as e:
        print("Error:", e)
        return [], []
    
def q_a_processing(text, df, tag, counter):
    """
    Input:
    text: the text to process using nltk
    df: the dataframe with all the information in it
    tag: the column name that will be used for filtering the dataframe
    counter: the int that keeps track of how many times we have processed a question
    ---
    Output:
    Tuple with the counter updated and the answer filtered
    ---
    Using the dataframe given by parameter look for the answer of the question given by tag
    """
    category = df.at[tag, 'category'].split(',')
    posible = df.at[tag, 'posible'].split(',')
    no_wanted = df.at[tag, 'no_wanted'].split(',')
    text = text.lower().replace(".","").replace("!","").replace("?","") # Remove punctuation
    etiquetas = nltk_processing(text)[1]
    possible_responses = []
    for tuple in etiquetas:
        if tuple[1] in category or tuple[0] in posible:
            possible_responses.append(tuple[0])
    response = [palabra for palabra in possible_responses if palabra.lower() not in no_wanted]
    if response == []: 
        answer = ""
        counter += 1
    else:
        answer = "".join(response)
        counter = 4
    return counter, answer

def q_a_gpt(client, question, transcription, counter):
    """
    Input:
    client: AzureOpenAI's client instance to process the question
    question: the question to be answered
    transcription: the transcription of the response
    counter: the int that keeps track of how many times we have processed a question
    ---
    Output:
    Tuple with the counter updated and the answer filtered
    ---
    Using the client given by parameter look for the answer of the question
    """
    prompt = """
        You are an assistant that processes transcribed audio responses. Your task is to interpret the context, correct any transcription errors, and provide the main corrected response concisely. If you cannot determine a logical response to the question from the given transcription, respond only with "Retry".

        Here are some examples of how your output should be according to the question and transcription.

        Examples:
        
        input: 
            Question: "What is your favourite drink?"
            Transcription: "My favorite drink is so that"
        expected output: "Soda"

        input: 
            Question: "How old are you?"
            Transcription: "I am twenty one years old"
        expected output: "21"

        input: 
            Question: "What is your name?"
            Transcription: "I call myself John"
        expected output: "John"

        input: 
            Question: "Where do you live?"
            Transcription: "I reside in New Yolk"
        expected output: "New York"

        input: 
            Question: "What is your favourite drink?"
            Transcription: "T"
        expected output: "Tea"

        input: 
            Question: "What is your favorite movie?"
            Transcription: "My favorite movie is The Lord of the Reigns"
        expected output: "The Lord of the Rings"

        """
    
    interaction = f"""
        Question: "{question}"
        Transcription: "{transcription}"
        
        output:
    """

    prediction = client.chat.completions.create(
        model="GPT-4o", 
        messages=[{"role":"system","content":prompt}, {"role":"user","content":interaction}], 
        temperature=0.5, 
        max_tokens=100
    )

    response = prediction.choices[0].message.content

    if response == "Retry": 
        answer = ""
        counter += 1
    else:
        answer = response.replace("output","").replace("expected","").replace(":","")
        counter = 4
    return counter, answer
    
def word_to_sec(text, wpm):
    """
    Input:
    text: the sentence for which you want to estimate the speaking time
    wpm: the words per minute speed to use for the estimate
    ---
    Output:
    the estimated time it would take to say the sentence in seconds
    ---
    Calculates the estimated time it would take to say a sentence at a given speed of words per minute (wpm) using experimental constants
    """
    c_1 = 37.6843059490085
    c_3 = 0.7
    tokens = text.split()
    n_tokens = len(tokens)
    time = ((n_tokens/wpm)*c_1) + c_3
    c_2 = (time/65)
    return time + c_2

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound
