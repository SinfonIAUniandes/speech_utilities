# This file contains the speech library class which is used to have all the functions related to the speech library
import os
import re
import soundfile as sf
import numpy as np
import nltk
import ConsoleFormatter
import rospkg
import openai
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
    model_size: size of the model to use ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large']
    ---
    Output:
    response of the local model with the transcription of the audio
    ---
    Use the local version of whisper for transcribing short audios
    """
    t1 = float(time.perf_counter() * 1000)
    result = model.transcribe(file_path)
    torch.cuda.empty_cache()
    t2 = float(time.perf_counter() * 1000)
    print("Local [ms]: ", float(t2-t1))
    return result["text"]

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
    return model_whisp

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

def auto_cut_function(speech_utilities):
    """
    Input:
    speech_utilities: speech_utilities object
    ---
    This function is used to automatically cut the audio when the person stops talking by changing the auto_finished variable to True
    """
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    if np.mean(np.abs(speech_utilities.audio_chunk)) > speech_utilities.silence_threshold and not speech_utilities.started_talking:
        speech_utilities.times_above_threshold += 2
    else:
        if speech_utilities.times_above_threshold > 0:
            speech_utilities.times_above_threshold -= 1
    if speech_utilities.times_above_threshold > 8 and not speech_utilities.started_talking:
        speech_utilities.started_talking = True
        setLedsColor(0,255,255)
        print(speech_utilities.started_talking)
        print("times above threshold",speech_utilities.times_above_threshold)
        print(consoleFormatter.format("Person started talking", "OKGREEN"))
    if np.mean(np.abs(speech_utilities.audio_chunk)) < speech_utilities.silence_threshold and speech_utilities.started_talking:
        speech_utilities.times_below_threshold += 1
    else:
        if speech_utilities.times_below_threshold > 0:
            speech_utilities.times_below_threshold -= 2
    if speech_utilities.times_below_threshold > 15 and not speech_utilities.auto_finished:
        setLedsColor(255,255,255)
        speech_utilities.auto_finished = True
        
def gpt(messages,temperature):
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
    openai.api_type="azure"
    openai.api_version = "2023-05-15"
    prediction = openai.ChatCompletion.create(
                api_key= os.getenv("GPT_API"),
                api_base="https://sinfonia.openai.azure.com/" ,
                engine="sinfoniaOpenai",
                temperature= temperature,
                max_tokens=100,
                messages = messages
            )
    return prediction['choices'][0]['message']

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
    c_3 = 0.5
    tokens = text.split()
    n_tokens = len(tokens)
    time = ((n_tokens/wpm)*c_1) + c_3
    c_2 = (time/65)
    return time + c_2
