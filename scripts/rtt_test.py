#!/usr/bin/env python3.11
import time
import rospkg
import rospy
import threading
import rosservice
import ConsoleFormatter
import os
import speech_library as sl
from RealtimeSTT import AudioToTextRecorder
import pyautogui
import struct
from openai import AzureOpenAI

# Speech_msgs
from speech_msgs.srv import speech2text_srv, answer_srv, q_a_srv, talk_srv, hot_word_srv

# Robot_msgs
from robot_toolkit_msgs.srv import audio_tools_srv, misc_tools_srv, set_speechrecognition_srv, set_output_volume_srv
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

        #self.listening = False
        self.listening = True
        self.sample_rate = 16000

        # Autocut variables
        self.times_below_threshold = 0
        self.times_above_threshold =0
        self.auto_finished = False
        self.speech_2_text_buffer = []
        self.started_talking = False


        # isTalking variable
        self.robot_speaking = False


        #Google
        self.person_speaking = False
        # 0 Es que recien le acaban de hablar, el numero se refiere a hace cuantos buffers fue la ultima instancia de habla
        self.last_speaking_instance = 0
        
        self.recorder = AudioToTextRecorder(use_microphone=False, spinner=False, language="es", model="base", silero_sensitivity=0.25, silero_deactivity_detection=True)
        self.process_text_thread = threading.Thread(target=self.recorder_to_text)
        self.process_text_thread.start()
        
        self.rospy_check = threading.Thread(target=self.check_rospy)
        self.rospy_check.start()

        
        # OpenAI GPT Model
        self.clientGPT = AzureOpenAI(
            azure_endpoint= "https://sinfonia.openai.azure.com/",
            api_key= os.getenv("GPT_API"),
            api_version="2024-02-01",
        )

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
            self.customSpeech.command = "custom"
            self.customSpeech.frequency = 16000
            self.customSpeech.channels = 3
            self.audioToolsService(self.customSpeech)

            # Enable mic
            self.turn_mic_pepper(True)

            # Publisher toolkit
            print(consoleFormatter.format("--Speech utilities Running in PEPPER--", "OKGREEN"))  

            # Connect to the Toolit for Hot Word detection
            print(consoleFormatter.format("Waiting for pytoolkit/ALSpeechRecognition/set_speechrecognition_srv...", "WARNING"))
            rospy.wait_for_service("/pytoolkit/ALSpeechRecognition/set_speechrecognition_srv")
            self.speech_recognition = rospy.ServiceProxy("/pytoolkit/ALSpeechRecognition/set_speechrecognition_srv", set_speechrecognition_srv)

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
        robot_name = "nova"
        print(consoleFormatter.format(f'ROBOT NAME IS {robot_name}', 'OKGREEN'))
        # Answer variables
        self.robot_name = robot_name.lower()
        self.info_herramientas = ""
        with open(self.PATH_DATA+"/tools_info.txt","r",encoding="utf-8") as file:
            self.info_herramientas = file.read()
        with open(self.PATH_DATA+f"/{self.robot_name}_info.txt","r",encoding="utf-8") as file:
            self.system_msg = file.read().replace("{info_herramientas}",self.info_herramientas)
        self.conversation_gpt = [{"role":"system","content":self.system_msg}]
            
            
        # ================================== SERVICES DECLARATION ==================================
            
        print(consoleFormatter.format('waiting for speech2text service!', 'WARNING'))  
        self.speech2text_declaration= rospy.Service("speech_utilities/speech2text_srv", speech2text_srv, self.callback_speech2text)
        print(consoleFormatter.format('speech2text on!', 'OKGREEN'))
        
        print(consoleFormatter.format('waiting for talk service!', 'WARNING'))  
        self.talk_declaration= rospy.Service("speech_utilities/talk_srv", talk_srv, self.callback_talk)
        print(consoleFormatter.format('talk on!', 'OKGREEN'))

        # ================================== SUBSCRIBER TO MIC DECLARATION ==================================

        self.micSubscriber=rospy.Subscriber("/mic", AudioBuffer, self.audioCallbackSingleChannel)
        
        
        self.speech_pub=rospy.Publisher('/speech', speech_msg, queue_size=10)

########################################  SPEECH SERVICES  ############################################
        
        
        
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
            self.talk(text,req.language,req.animated,wait=req.wait)
        return f"Pepper said: {req.key}"
    
    def talk(self,key,language,animated,wait):
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
        if wait:
            t1 = float(time.perf_counter() * 1000)
            timeout = sl.word_to_sec(key, 100)
            self.robot_speaking=True
            while self.robot_speaking:
                rospy.sleep(0.05)
                elapsed = float(time.perf_counter() * 1000)
                if (float(elapsed-t1))/1000 >= timeout:
                    pass
            self.robot_speaking=False
        key = key.replace("\\rspd=100\\","")
        print(consoleFormatter.format(f"Pepper said: {key}","OKGREEN"))
        
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
            # Send the command to the audio service
            
            self.customSpeech = audio_tools_msg()
            self.customSpeech.command = "custom"
            self.customSpeech.frequency = 16000
            self.customSpeech.channels = 3
            self.audioToolsService(self.customSpeech)
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
        #Set led color to blue
        sl.setLedsColor(0,255,255)
        rospy.sleep(1)
        # If the duration is 0, the recording will be stopped when the person stops talking
        if duration == 0:
            # Timeout if the person talking is not recognized or it takes too long
            max_timeout = 20
            t1 = time.time()
            while not self.person_speaking and time.time()-t1<5:
                rospy.sleep(0.1)
            print(consoleFormatter.format("Person started talking", "OKGREEN"))
            while (self.person_speaking or self.last_speaking_instance < 30) and time.time()-t1<max_timeout:
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
        self.auto_finished = False
        self.started_talking = False
        self.set_volume(70)
        # Save the audio from the speech2text buffer
        sl.save_recording(self.speech_2_text_buffer,"speech2text",self.sample_rate)
        if lang=="esp":
            # Transcribe the audio
            transcription = sl.transcribe_spanish(self.PATH_DATA+"/speech2text.wav", self.google_recognize)
        else:
            # Transcribe the audio
            transcription = sl.transcribe(self.PATH_DATA+"/speech2text.wav", self.whisper_model) # ANTES
            #transcription = sl.transcribe_cloud(self.PATH_DATA+"/speech2text.wav", self.clientGPT) # Ahora se transcribe cloud con Azure
        self.speech_2_text_buffer = []
        self.toggle_blinking(True)
        self.listening = False
        print(consoleFormatter.format(f"Local listened: {transcription}", "OKGREEN"))
        return transcription
    

    # ================================== PEPPER AUDIO  ==================================
    
    def check_rospy(self):
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
        print(consoleFormatter.format("Shutting down", "FAIL"))
        os._exit(os.EX_OK)
        
    def audioCallbackSingleChannel(self, data):
        """
        Callback function for the /mic topic
        data.data: audio buffer (PCM raw data, signed 16-bit)
        """
        if self.listening:
            
            # Check if the incoming data is a tuple of signed 16-bit PCM values
            if isinstance(data.data, tuple):
                try:
                    # Pack the tuple of signed 16-bit PCM values into a bytearray
                    data.data = bytearray(struct.pack(f'{len(data.data)}h', *data.data))
                except struct.error as e:
                    print(f"Error packing PCM data into bytearray: {e}")
            
            self.speech_2_text_buffer.extend(data.data)
            self.recorder.feed_audio(chunk=data.data)

    def process_text(self,text):
        print(text + " ")
        request = f"""La persona dijo: {text}. Si hay palabras en otro idioma en tu respuesta escribelas como se pronunicarian en español porque en este momento solo puedes hablar español y ningun otro idioma, por ejemplo si en tu respuesta esta Python, responde Paiton. No añadas contenido complejo a tu respuesta como codigo, solo explica lo que sea necesario. Manten tus respuestas cortas"""
        self.conversation_gpt.append({"role":"user","content":request})
        response = sl.gpt(self.clientGPT, self.conversation_gpt,0)["content"]
        self.listening = False
        self.talk(key=response,language="Spanish",animated=False,wait=True)
        self.listening = True
    
    def recorder_to_text(self):
        while True:
            self.recorder.text(self.process_text)

    # ================================== ROBOT ISTALKING ==================================
    def callback_check_speaking(self,data):
        """
        Callback function for the /pytoolkit/ALTextToSpeech/status topic
        data.data (str): string saying if the robot is speaking
        """
        if data.status == "done":
            self.robot_speaking = False
        else:
            self.robot_speaking = True

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
