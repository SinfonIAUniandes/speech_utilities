# This file contains the speech library class which is used to have all the functions related to the speech library
import os
import re
import soundfile as sf
import numpy as np
import ConsoleFormatter
import rospkg
import openai
import rospy    
# === Import messages ===
from robot_toolkit_msgs.msg import leds_parameters_msg

import time

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
        print(speech_utilities.started_talking)
        print("times above threshold",speech_utilities.times_above_threshold)
        print(consoleFormatter.format("Person started talking", "OKGREEN"))
    if np.mean(np.abs(speech_utilities.audio_chunk)) < speech_utilities.silence_threshold and speech_utilities.started_talking:
        speech_utilities.times_below_threshold += 1
    else:
        if speech_utilities.times_below_threshold > 0:
            speech_utilities.times_below_threshold -= 2
    if speech_utilities.times_below_threshold > 15 and not speech_utilities.auto_finished:
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
    
def word_to_sec(text, wpm):
    """API
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


        