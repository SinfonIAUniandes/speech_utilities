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

        #Google
        self.person_speaking = False
        # 0 Es que recien le acaban de hablar, el numero se refiere a hace cuantos buffers fue la ultima instancia de habla
        self.last_speaking_instance = 0
        
        self.recorder = AudioToTextRecorder(use_microphone=False, spinner=False)
        self.process_text_thread = threading.Thread(target=self.recorder_to_text)
        self.process_text_thread.start()
        
        self.rospy_check = threading.Thread(target=self.check_rospy)
        self.rospy_check.start()

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
            
            #Set led color to white
            sl.setLedsColor(255,255,255)
            
            
        # ================================== SERVICES DECLARATION ==================================
            
        print(consoleFormatter.format('waiting for speech2text service!', 'WARNING'))  
        self.speech2text_declaration= rospy.Service("speech_utilities/speech2text_srv", speech2text_srv, self.callback_speech2text)
        print(consoleFormatter.format('speech2text on!', 'OKGREEN'))

        # ================================== SUBSCRIBER TO MIC DECLARATION ==================================

        self.micSubscriber=rospy.Subscriber("/mic", AudioBuffer, self.audioCallbackSingleChannel)

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
        pyautogui.typewrite(text + " ")
    
    def recorder_to_text(self):
        while True:
            self.recorder.text(self.process_text)

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
