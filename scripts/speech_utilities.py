#!/usr/bin/env python3.8
import time
import rospkg
import rospy
import numpy as np
import pandas as pd
import threading
import rosservice
import subprocess
import ConsoleFormatter
import sounddevice
import os
import speech_library as sl
import speech_recognition as sr
from openai import AzureOpenAI
import torch
torch.set_num_threads(1)

repo_or_dir = 'snakers4/silero-vad'
model_name = 'silero_vad'
model_dir = torch.hub.get_dir()

model_exists = False
try:
    vad_model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, force_reload=False)
    model_exists = True
except Exception as e:
    print(f"Modelo no encontrado localmente, descargando: {e}")

if not model_exists:
    vad_model, utils = torch.hub.load(repo_or_dir=repo_or_dir, model=model_name, force_reload=True)

# Speech_msgs
from speech_msgs.srv import speech2text_srv, answer_srv, calibrate_srv, q_a_srv, talk_srv, hot_word_srv

# Robot_msgs
from robot_toolkit_msgs.srv import audio_tools_srv, misc_tools_srv, set_speechrecognition_srv, set_words_threshold_srv, set_output_volume_srv, battery_service_srv
from robot_toolkit_msgs.msg import audio_tools_msg, speech_msg, text_to_speech_status_msg, misc_tools_msg, text_to_speech_status_msg

from std_srvs.srv import SetBool

# Naoqi_msgs
from naoqi_bridge_msgs.msg import AudioBuffer

class SpeechUtilities:

    # ===================================================== INIT ==================================================================

    def __init__(self):
        
        try:
            available_services = rosservice.get_service_list()
            self.ROS=True
        except:
            self.ROS=False

        # ================================== GLOBAL VARIABLES ==================================
        
        self.PATH_SPEECH_UTILITIES = rospkg.RosPack().get_path('speech_utilities')
        self.PATH_DATA = self.PATH_SPEECH_UTILITIES+'/data'

        self.listening = False
        self.audio_chunk =[]
        self.audio_buffer = []
        self.sample_rate =200000
        self.s2t =False

        # Autocut variables
        self.auto_cut = False
        self.times_below_threshold = 0
        self.times_above_threshold =0
        self.constant_silence_threshold = 3700
        self.silence_threshold = self.constant_silence_threshold
        self.auto_finished = False
        self.speech_2_text_buffer = []
        self.started_talking = False

        # Answer variables
        self.conversation_gpt = [{"role":"system","content":"You are a Pepper robot named Nova from the University of the Andes in Bogotá, Colombia, specially from the research group SinfonIA, you serve as a Social Robot and you are able to perform tasks such as guiding, answering questions, recognizing objects, people and faces, among others.You were built by SoftBank Robotics in France in 2014. You have been in the University since 2020 and you spend most of your time in Colivri laboratory. Answer all questions in the most accurate but nice way possible. "}]

        # isTalking variable
        self.isTalking = False

        # Whisper Model
        self.whisper_model = sl.load_model("small.en")
        #Google
        self.google_recognize = sr.Recognizer()
        self.person_speaking = False
        # 0 Es que recien le acaban de hablar, el numero se refiere a hace cuantos buffers fue la ultima instancia de habla
        self.last_speaking_instance = 0
        
        self.rospy_check = threading.Thread(target=self.check_rospy)
        self.rospy_check.start()

        
        # OpenAI GPT Model
        self.clientGPT = AzureOpenAI(
            azure_endpoint= "https://sinfonia.openai.azure.com/",
            api_key= os.getenv("GPT_API"),
            api_version="2024-02-01",
        )

        # ================================== IF LOCAL ==================================
            
        if not self.ROS or '/robot_toolkit/audio_tools_srv' not in available_services:
            # Initialice roscore
            subprocess.Popen('roscore')
            rospy.sleep(2)
            rospy.init_node('SpeechUtilities', anonymous=True)
            # Initialice local audio publisher from PC mic
            self.audio_pub=rospy.Publisher('/mic', AudioBuffer, queue_size=10)
            local_audio = threading.Thread(target=self.publish_local_audio)
            local_audio.start()
            print(consoleFormatter.format("--Speech utilities Running in LOCALLY--", "OKGREEN"))

        # ================================== IF PEPPER AVAILABLE ==================================

        if self.ROS:
            # Initialice speech node
            rospy.init_node('SpeechUtilities', anonymous=True)
            # Enable speech from toolkit
            rospy.wait_for_service('/robot_toolkit/audio_tools_srv')
            self.audioToolsService = rospy.ServiceProxy('/robot_toolkit/audio_tools_srv', audio_tools_srv)
            self.enableSpeech = audio_tools_msg()
            self.enableSpeech.command = "enable_tts"
            self.audioToolsService(self.enableSpeech)

            # Custom speech parameters for robot
            self.customSpeech = audio_tools_msg()
            self.customSpeech.command = "set_speech_params"
            self.customSpeech.speech_parameters.pitch_shift=1 # Grueso (1) o Agudo (2)
            self.customSpeech.speech_parameters.double_voice_level= 0.0
            self.customSpeech.speech_parameters.double_voice_time_shift= 0.0
            self.customSpeech.speech_parameters.speed= 120.0 # Velocidad al hablar
            self.audioToolsService(self.customSpeech)

            # Enable mic
            self.turn_mic_pepper(True)

            # Publisher toolkit
            print(consoleFormatter.format("--Speech utilities Running in PEPPER--", "OKGREEN"))  

            # Connect to the Toolit for Hot Word detection
            print(consoleFormatter.format("Waiting for pytoolkit/ALSpeechRecognition/set_speechrecognition_srv...", "WARNING"))
            rospy.wait_for_service("/pytoolkit/ALSpeechRecognition/set_speechrecognition_srv")
            self.speech_recognition = rospy.ServiceProxy("/pytoolkit/ALSpeechRecognition/set_speechrecognition_srv", set_speechrecognition_srv)

            print(consoleFormatter.format("Waiting for pytoolkit/ALSpeechRecognition/set_words_srv...", "WARNING"))
            rospy.wait_for_service("/pytoolkit/ALSpeechRecognition/set_words_srv")
            self.set_words = rospy.ServiceProxy("/pytoolkit/ALSpeechRecognition/set_words_srv", set_words_threshold_srv)

            print(consoleFormatter.format("Waiting for pytoolkit/ALAudioDevice/set_output_volume_srv...", "WARNING"))
            rospy.wait_for_service("/pytoolkit/ALAudioDevice/set_output_volume_srv")
            self.set_volume = rospy.ServiceProxy("/pytoolkit/ALAudioDevice/set_output_volume_srv", set_output_volume_srv)

            print(consoleFormatter.format("Waiting for pytoolkit/ALAutonomousBlinking/toggle_blinking_srv...", "WARNING"))
            rospy.wait_for_service("/pytoolkit/ALAutonomousBlinking/toggle_blinking_srv")
            self.toggle_blinking = rospy.ServiceProxy("/pytoolkit/ALAutonomousBlinking/toggle_blinking_srv", SetBool)

            # Subscriber Service Speech Recognition
            print(consoleFormatter.format('Waiting for /pytoolkit/ALTextToSpeech/status...', 'WARNING'))
            self.talkinSubscriber = rospy.Subscriber('/pytoolkit/ALTextToSpeech/status', text_to_speech_status_msg, self.callback_check_speaking)
            
            #Set led color to white
            sl.setLedsColor(255,255,255)
            
            
        # ================================== SERVICES DECLARATION ==================================
            
        print(consoleFormatter.format('waiting for speech2text service!', 'WARNING'))  
        self.speech2text_declaration= rospy.Service("speech_utilities/speech2text_srv", speech2text_srv, self.callback_speech2text)
        print(consoleFormatter.format('speech2text on!', 'OKGREEN'))
        
        print(consoleFormatter.format('waiting for answers_srv service!', 'WARNING'))  
        self.chatGPT_question_answer= rospy.Service("speech_utilities/answers_srv", answer_srv , self.callback_gpt_question_answer)
        print(consoleFormatter.format('answers_srv on!', 'OKGREEN'))

        print(consoleFormatter.format('waiting for hot_word_srv service!', 'WARNING'))  
        self.hot_word_declaration= rospy.Service("speech_utilities/hot_word_srv", hot_word_srv , self.callback_hot_word_srv)
        print(consoleFormatter.format('hot_word_srv on!', 'OKGREEN'))

        print(consoleFormatter.format('waiting for calibrate_srv service!', 'WARNING'))  
        self.calibrate_declaration= rospy.Service("speech_utilities/calibrate_srv", calibrate_srv , self.callback_calibrate)
        print(consoleFormatter.format('calibrate_srv on!', 'OKGREEN'))

        print(consoleFormatter.format('waiting for talk service!', 'WARNING'))  
        self.talk_declaration= rospy.Service("speech_utilities/talk_srv", talk_srv, self.callback_talk)
        print(consoleFormatter.format('talk on!', 'OKGREEN'))
        
        print(consoleFormatter.format('waiting for q_a service!', 'WARNING'))  
        self.q_a_declaration= rospy.Service("speech_utilities/q_a_srv", q_a_srv, self.callback_q_a)
        print(consoleFormatter.format('q_a on!', 'OKGREEN'))

        # ================================== SUBSCRIBER TO MIC DECLARATION ==================================

        self.micSubscriber=rospy.Subscriber("/mic", AudioBuffer, self.audioCallbackSingleChannel)

        # ================================== PUBLISHER TO SPEECH DECALRATION ==================================

        self.speech_pub=rospy.Publisher('/speech', speech_msg, queue_size=10)
        
        # ================================== SERVICE PROXY ==================================

########################################  SPEECH SERVICES  ############################################
        
    # ================================== TURN MIC PEPPER ==================================
    def turn_mic_pepper(self, enable):
        """
        Input:
        bool enable: if true, the microphone will be enabled. If false, the microphone will be disabled
        ---
        Output:
        bool: if true, the microphone was successfully enabled or disabled. If false, there was an error
        ---
        Enables or disables the microphone of the robot
        """

        command = "enable" if enable else "disable"

        try:
            misc = rospy.ServiceProxy('/robot_toolkit/misc_tools_srv', misc_tools_srv)

            miscMessage = misc_tools_msg()
            miscMessage.command = "enable_all"
            misc(miscMessage)

            rospy.wait_for_service('/robot_toolkit/audio_tools_srv')
            audio = rospy.ServiceProxy('/robot_toolkit/audio_tools_srv', audio_tools_srv)
            # Send the command to the audio service
            audioMessage = audio_tools_msg()
            audioMessage.command = "disable" if enable else "enable"
            audio(audioMessage)
            audioMessage = audio_tools_msg()
            audioMessage.command = command
            rospy.sleep(1)
            audio(audioMessage)
            return True
        
        except rospy.ServiceException as e:
            print(f"Error al cambiar el estado del micrófono: {e}")
            return False
        

  
    # ================================== SPEECH2TEXT ==================================
    
    def callback_speech2text(self, req):
        """
        Input:
        int32 duration: duration of the recording in seconds. If 0, the recording will be stopped when the person stops talking
        ---
        Output: 
        string transcription: transcription of the audio
        ---
        Returns the transcription of the audio from the microphone
        """
        transcription = self.speech2text(req.duration, req.lang)
        return transcription
    
    def speech2text(self, duration, lang=""):
        """
        Output: 
        string transcription: transcription of the audio
        ---
        Returns the transcription of the audio from the microphone
        """
        print(consoleFormatter.format("Requested sppech2text service!", "OKGREEN"))
        self.toggle_blinking(False)
        # Initialize a special buffer for the speech2text
        self.set_volume(0)
        self.speech_2_text_buffer = []
        #Set eyes to blue
        self.listening = True
        self.s2t = True
        if lang=="esp":
            audio_tools_proxy = rospy.ServiceProxy('/robot_toolkit/audio_tools_srv', audio_tools_srv)
            audioMessage = audio_tools_msg()
            audioMessage.command = "custom"
            audioMessage.frequency = 16000
            audioMessage.channels = 3
            audio_tools_proxy(audioMessage)
            rospy.sleep(1)
        #Set led color to blue
        sl.setLedsColor(0,255,255)
        rospy.sleep(1)
        # If the duration is 0, the recording will be stopped when the person stops talking
        if duration == 0:
            # Timeout if the person talking is not recognized or it takes too long
            max_timeout = 20
            t1 = time.time()
            self.auto_cut = False
            while not self.person_speaking and time.time()-t1<5:
                rospy.sleep(0.1)
            print(consoleFormatter.format("Person started talking", "OKGREEN"))
            while (self.person_speaking or self.last_speaking_instance < 13) and time.time()-t1<max_timeout:
                rospy.sleep(0.1)
                t1 = time.time()
            if time.time()-t1>=max_timeout:
                print(consoleFormatter.format("Timeout reached", "FAIL"))
            else:
                print(consoleFormatter.format("Person finished talking", "OKGREEN"))
        # If the duration is not 0, the recording will be stopped after the duration
        else:
            rospy.sleep(duration)
        #Set led color to white
        sl.setLedsColor(255,255,255)
        self.s2t = False
        self.auto_cut = False
        self.auto_finished = False
        self.started_talking = False
        self.set_volume(70)
        if lang=="esp":
            # Save the audio from the speech2text buffer
            sl.save_recording(self.speech_2_text_buffer,"speech2text",16000)
            # Transcribe the audio
            transcription = sl.transcribe_spanish(self.PATH_DATA+"/speech2text.wav", self.google_recognize)
            audioMessage = audio_tools_msg()
            audioMessage.command = "disable"
            audio_tools_proxy(audioMessage)
            audioMessage = audio_tools_msg()
            audioMessage.command = "enable"
            rospy.sleep(1)
            audio_tools_proxy(audioMessage)
        else:
            # Save the audio from the speech2text buffer
            sl.save_recording(self.speech_2_text_buffer,"speech2text",self.sample_rate)
            # Transcribe the audio
            # transcription = sl.transcribe(self.PATH_DATA+"/speech2text.wav", self.whisper_model) # ANTES
            transcription = sl.transcribe_cloud(self.PATH_DATA+"/speech2text.wav", self.clientGPT) # Ahora se transcribe cloud con Azure
        self.speech_2_text_buffer = []
        self.toggle_blinking(True)
        print(consoleFormatter.format(f"Local listened: {transcription}", "OKGREEN"))
        return transcription
    
    # ================================== HOT WORD ==================================
    def callback_hot_word_srv(self, req):
        """
        Input:
        list of string hot_words: list of hot words to detect
        eyes: if true, the eyes will be activated
        sound: if true, the sound will be activated
        threshold: threshold to detect the hot words
        ---
        Output:
        bool: if true, the hot word service started publishing the hot words. If false, the service was turned off or there is no Toolkit
        ---
        Starts the hot word service, when it detects the hot word, it will publish the hot word in the /pytoolkit/ALSpeechRecognition/status topic
        """
        response = False
        print(consoleFormatter.format("Requested hot word service!", "OKGREEN"))
        if self.ROS:
            hot_words = req.hot_words
            threshold = req.thresholds
            if hot_words == []:
                print(consoleFormatter.format("Turning off the hot_word_srv", "FAIL"))
                self.speech_recognition(False,False,False)
            else:
                print(consoleFormatter.format(f"Detecting words: {hot_words}, with threshold: {threshold}", "OKGREEN"))
                self.speech_recognition(True,req.noise,req.eyes)
                self.set_words(hot_words,threshold)
                response = True
        else:
            print(consoleFormatter.format("No Toolkit available", "FAIL"))
        return response

    # ================================== CALIBRATE ==================================
    def callback_calibrate(self, req):
        """
        Input:
        int32 duration: duration of the recording in seconds
        ---
        Output:
        float64 threshold: silence threshold
        ----------
        Returns the silence threshold of the audio from the microphone
        """
        print(consoleFormatter.format("Requested calibrate service!", "OKGREEN"))
        # Empty the audio buffer
        self.audio_buffer = []
        self.listening = True
        rospy.sleep(req.duration)
        self.listening = False
        buffer = self.audio_buffer
        # Calculate the silence threshold, adjust the weight for the specific case for maximum amplitude and absolute mean (0.85*mean+0.15*max suggested)
        calibrated_silence_threshold = round((np.mean(np.abs(buffer))*0.9+np.max(buffer)*0.1),2)
        # calibrated_silence_threshold = np.mean(np.abs(buffer))
        self.silence_threshold = 1500
        # self.silence_threshold = round((calibrated_silence_threshold+self.constat_silence_threshold)/2,2)
        print(consoleFormatter.format(f'Constant threshold: {self.constant_silence_threshold} \n Calibrated threshold: {calibrated_silence_threshold} \n Silence threshold : {self.silence_threshold}', 'OKBLUE'))
        return self.silence_threshold

    # ================================== TALK ==================================
    def callback_talk(self, req):
        """
        Input:
        string key: Indicates the phrase that the robot must say
        string language: Indicates the language which Pepper will speak [English, Spanish]
        bool wait: Indicates if the robot should wait to shoot down the service
        bool animated: Indicates if the robot should make gestures while talking
        string talk_speed: Indicates the speech speed the robot will talk between 50-400 (default: 100)
        ---
        Callback for talk_speech_srv service: This service allows the robot to say the input of the service
        """
        print(consoleFormatter.format("Requested talk service!", "OKGREEN"))
        if req.talk_speed == "":
            req.talk_speed = 100
        if self.ROS:
            text = f"\\rspd={req.talk_speed}\\{req.key}"
            self.talk(text,req.language,req.animated)
        print(consoleFormatter.format("Talking...","WARNING"))
        if req.wait:
            t1 = float(time.perf_counter() * 1000)
            timeout = sl.word_to_sec(req.key, float(req.talk_speed))
            self.isTalking=True
            while self.isTalking:
                rospy.sleep(0.05)
                elapsed = float(time.perf_counter() * 1000)
                if (float(elapsed-t1))/1000 >= timeout:
                    break
            self.isTalking=False
        print(consoleFormatter.format(f"Pepper said: {req.key}","OKGREEN"))
        return f"Pepper said: {req.key}"
    
    def talk(self,key,language,animated):
        """
        Input:
        string key: Indicates the phrase that the robot must say
        string language: Indicates the language which Pepper will speak [English, Spanish]
        bool animated: Indicates if the robot should make gestures while talking
        string talk_speed: Indicates the speech speed the robot will talk between 50-400 (default: 100)
        ---
        Internal function for speech services: This function allows the robot to say the input of the service. It halts execution.
        """
        t2s_msg = speech_msg()
        t2s_msg.animated = animated
        t2s_msg.language = language
        t2s_msg.text = key
        self.speech_pub.publish(t2s_msg)
        print(consoleFormatter.format("Talking...","WARNING"))
        t1 = float(time.perf_counter() * 1000)
        timeout = sl.word_to_sec(key, 100)
        self.isTalking=True
        while self.isTalking:
            rospy.sleep(0.05)
            elapsed = float(time.perf_counter() * 1000)
            if (float(elapsed-t1))/1000 >= timeout:
                break
        self.isTalking=False
        key = key.replace("\\rspd=100\\","")
        print(consoleFormatter.format(f"Pepper said: {key}","OKGREEN"))
        
    
    # ================================== Q_A ==================================
    def callback_q_a(self, req):
        """
        Input:
        string tag: key word for the question that the robot will say depending on .xlsx file, always in lowercase [name, age, drink, gender]
        ---
        Callback for q_a_speech_srv, this service return a specific answer for predefined questions
        """
        print(consoleFormatter.format("Requested Q&A service!", "OKGREEN"))
        df = pd.read_csv(self.PATH_DATA+'/data.csv')
        tags = df['tag'].tolist()
        df.set_index('tag', inplace=True)
        if req.tag in tags:
            question_value = df.at[req.tag, 'question']
        else:
            print(consoleFormatter.format(f"Invalid Tag", "FAIL"))
            return "Invalid Tag"
        counter = 0
        while counter < 3:
            self.talk(question_value, "English", False)
            rospy.sleep(1)
            text = self.speech2text(0)
            if req.tag=="drink":
                if ("so that" in text or "so then" in text or "solar" in text or "so" in text):
                    text = "soda"
                elif ("cock" in text):
                    text = "coke"
            print(f"Transcription: {text}")
            # counter, answer = sl.q_a_processing(text, df, req.tag, counter) # ANTES
            counter, answer = sl.q_a_gpt(self.clientGPT, question_value, text, counter) # Ahora se procesa con GPT-4o
        print(consoleFormatter.format(f"Local listened: {answer}", "OKGREEN"))
        return answer

    # ================================== GPT Q&A ==================================
    def callback_gpt_question_answer(self, req):
        """
        Input:
        string question: question to ask
        bool save_conversation: if true, the conversation will be saved and the model will answer regarding previous questions
        float64 temperature: (0-1) the higher the temperature, the more random the answer
        system_msg: message to be added to the content of system in the conversation
        ---
        Output:
        string answer: answer to the question
        """
        print(consoleFormatter.format("Requested answer service!", "OKGREEN"))
        print(consoleFormatter.format(f"Question: {req.question}", "WARNING"))
        if not req.save_conversation:
            system_msg = f"""You are a Pepper robot named Nova from the University of the Andes in Bogotá, Colombia, 
            specially from the research group SinfonIA, you serve as a Social Robot and you are able 
            to perform tasks such as guiding, answering questions, recognizing objects, people and faces, among others.
            You were built by SoftBank Robotics in France in 2014. You have been in the University since 2020 and you spend most of your time in Colivri laboratory.
            Answer all questions in the most accurate but nice way possible. {req.system_msg}""" 
            self.conversation_gpt = [{"role":"system","content":system_msg}]
        self.conversation_gpt.append({"role":"user","content":req.question})
        response = sl.gpt(self.clientGPT, self.conversation_gpt,req.temperature)
        if "content" in response:
            answer = response["content"]
            print(consoleFormatter.format(f"Response: {answer}", "OKBLUE"))
            self.conversation_gpt.append(response)
            if answer== "": 
                answer = "I could not find relevant results for your question "
        else:
            answer = "I could not find relevant results for your question "
        return answer

    # ================================== LOCAL AUDIO  SETUP ==================================
    def publish_local_audio(self):
        """
        Extracts the audio from the local microphone
        """
        with sounddevice.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sample_rate, blocksize= 16384):
            while not rospy.is_shutdown():
                rospy.sleep(0.01)
            sounddevice.stop()
            subprocess.Popen('killall -9 roscore rosmaster')

    def audio_callback(self, indata, frames, time, status): 
        """
        Publishes the audio from the local microphone into the fake /mic topic
        """
        audio_msg = AudioBuffer()
        audio_data = (indata * 32767).astype(np.int16)
        audio_msg.data = audio_data.flatten().tolist()
        self.audio_pub.publish(audio_msg)

    # ================================== PEPPER AUDIO  ==================================
    
    def check_rospy(self):
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
        print(consoleFormatter.format("Shutting down", "FAIL"))
        os._exit(os.EX_OK)
        
    def audioCallbackSingleChannel(self, data):
        """
        Callback function for the /mic topic
        data.data: audio buffer
        """
        # If the listening variable is set to True, the audio will be saved constantly in the audio buffer
        if self.listening:
            self.audio_chunk = list(data.data)
            self.audio_buffer.extend(data.data)
            # When the speech2text is enabled, the variable self.speech_2_text_buffer will be filled with the audio data
            if self.s2t:
                self.speech_2_text_buffer.extend(data.data)
        audio_data = np.array(data.data, dtype=np.int16)
        audio_int16 = np.frombuffer(audio_data.tobytes(), np.int16);
        audio_float32 = sl.int2float(audio_int16)
        new_confidence = vad_model(torch.from_numpy(audio_float32), 48000).item()
        if new_confidence>0.6:
            self.person_speaking = True
            self.last_speaking_instance = 0
        else:
            self.person_speaking = False
            self.last_speaking_instance += 1

    # ================================== PEPPER ISTALKING ==================================
    def callback_check_speaking(self,data):
        """
        Callback function for the /pytoolkit/ALTextToSpeech/status topic
        data.data (str): string saying if the robot is speaking
        """
        if data.status == "done":
            self.isTalking = False
        else:
            self.isTalking = True
# ================================== MAIN ==================================

if __name__ == '__main__':
    consoleFormatter=ConsoleFormatter.ConsoleFormatter()
    print(consoleFormatter.format(" --- trying to initialize speech utilities node ---","OKGREEN"))
    speechUtilities = SpeechUtilities()
    try:
        if speechUtilities.ROS:
            print(consoleFormatter.format(" --- PEPPER speech utilities node successfully initialized ---","OKGREEN"))
        else:
            print(consoleFormatter.format(" --- LOCAL speech utilities node successfully initialized ---","OKGREEN"))
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
