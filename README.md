# Speech Utilities

This package offers a series of ROS services that help the robot to record an audio, convert it to text and make some basic questions and answers for the other tools. We are part of SinfonIA Uniandes

**Table of Contents**

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Dependencies](#dependencies)
    - [Libraries](#libraries)
    - [ROS Packages](#ros-packages)
  - [Install](#install)
- [Execution](#execution)
- [Usage](#usage)
  - [Services](#services)
    - [talk_speech_srv](#talk_speech_srv)
    - [q_a_speech_srv](#q_a_speech_srv)
    - [conversation_srv](#conversation_srv)

# Installation

## Requirements

- Linux Ubuntu 20.04
- ROS Melodic/Noetic
- Python >= 3.8

## Dependencies

### Libraries

First of all, you must run these commands on the terminal.

```bash
sudo apt update && sudo apt install ffmpeg
sudo apt-get install portaudio19-dev
sudo apt install ffmpeg
```

Next, to install everything use pip install -r requirements.txt. That will install the following libraries:

- ffmpeg==1.4
- git+https://github.com/openai/whisper.git
- nltk==3.8.1
- openai==1.2.4
- openpyxl==3.1.2
- pandas==2.0.2
- sounddevice==0.4.6
- soundfile==0.12.1
- spacy==3.5.2
- SpeechRecognition==3.10.0
- vosk==0.3.45

### ROS Packages

These packages should be on the src of the workspace.

- audio_common_msgs [audio_common_msgs]

```bash
 git clone https://github.com/ros-drivers/audio_common.git
```

- naoqi_bridge_msgs [naoqi_bridge_msgs]

```bash
 git clone https://github.com/ros-naoqi/naoqi_bridge_msgs.git
```

- robot_toolkit_msgs [robot_toolkit_msgs]

```bash
 git clone https://github.com/SinfonIAUniandes/robot_toolkit_msgs.git
```

## Install

1.  Clone the repositories (speech_utilities and speech_msgs) to the src folder of the workspace (in the same folder that the other ROS Packages).

```bash
  git clone https://github.com/SinfonIAUniandes/speech_utilities.git
  git clone https://github.com/SinfonIAUniandes/speech_msgs.git
```

2.  Move to the root of the workspace and build the workspace.

```bash
  cd ..
  catkin_make
  source devel/setup.bash
```

# Execution

When roscore is available run:

```bash
  rosrun speech_utilities speech_utilities.py
```

# Usage

## Services

Speech_unite offers the following services:

### talk_srv

- **Description:**
  This service allows the robot to say the input of the service.
- **Service file:** _talk_srv.srv_
  - **Request:**
    - key (string): Indicates the phrase that the robot must say.
    - language (string): Indicates the language which robot will speak. Could be 'English' or 'Spanish'.
    - wait (bool): Indicates if the robot should wait to shoot down the service.
    - animated (bool): Indicates if the robot should make gestures while talking.
    - talk_speed (string): Indicates the speech speed the robot will talk between 50-400 (default: 100).
  - **Response**:
    - result (string): Indicates what the robot is talking.
- **Call service example:**

```bash
 rosservice call /speech_utilities/talk_srv "key: 'Hello my name is Nova.'  language: 'English' wait: false animated: false talk_speed: '85'"
  ```

---

### q_a_speech_srv

- **Description**
  This service allows the robot to say some questions pre established, start recording the audio throw the save_audio_srv and return an answer with the whisper and data.xlsx loaded in data folder.
- **Service file:** _q_a_speech_srv.srv_
  - **Request:**
    - tag (string): Indicates the key word for the question that the robot will say. For example: 'birth' if for 'When is your birthday?'. Allowed keys: name, age, drink, gender. Must be in lowercase.
  - **Response:**
    - answer (string): Indicates what Pepper ask for (the question).
- **Call service example:**

```bash
 rosservice call /speech_utilities/q_a_speech_srv "tag: 'age'"
```

---

### speech2text_srv

- **Description**
  This service allows the robot to returns the transcription of the audio from the microphone.
- **Service file:** _speech2text_srv.srv_
  - **Request:**
    - duration (int32): Duration of the recording in seconds. If 0, the recording will be stopped when the person stops talking.
  - **Response:**
    - transcription (string): Transcription of the audio.
- **Call service example:**

```bash
 rosservice call /speech_utilities/speech2text_srv "duration: 0"
```

---

### calibrate_srv

- **Description**
  Returns the silence threshold of the audio from the microphone.
- **Service file:** _calibrate_srv.srv_
  - **Request:**
    - duration (int32): Duration of the recording in seconds.
  - **Response:**
    - threshold (float64): Silence threshold.
- **Call service example:**

```bash
 rosservice call /speech_utilities/calibrate_srv "duration: 5"
```

---

### answer_srv

- **Description**
  This service allows the robot to answer a question using a OpenAI model.
- **Service file:** _answer_srv.srv_
  - **Request:**
    - question (string): Indicates the question to solve.
    - save_conversation (bool): If true, the conversation will be saved and the model will answer regarding previous questions.
    - temperature (float64): (0-1) the higher the temperature, the more random the answer.
    - system_msg (string): Message to be added to the content of system in the conversation.
  - **Response:**
    - answer (string): Indicates the answer of the question.
- **Call service example:**

```bash
 rosservice call /speech_utilities/answer_srv "question: 'Who discover America?' language: 'en'"
```

---

### hot_word_srv

- **Description**
  This service allows the robot to answer a question using a Google API or a own model.
- **Service file:** _hot_word_srv.srv_
  - **Request:**
    - hot_words (list[String]): List of hot words to detect.
    - eyes (bool): If true, the eyes will be activated.
    - sound (bool): If true, the sound will be activated.
    - threshold (float64): Threshold to detect the hot words.
  - **Response:**
    - response (bool): If true, the hot word service started publishing the hot words. If false, the service was turned off or there is no Toolkit.
- **Call service example:**

```bash
 rosservice call /speech_utilities/hot_word_srv "hot_words: ['palabra1', 'palabra2', 'palabra3'] noise: false eyes: true threshold: 0.5"

```

---

[audio_common_msgs]: https://github.com/ros-drivers/audio_common/tree/master/audio_common_msgs "audio_common_msgs"
[DeepSpeech_model]: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite "DeepSpeech_model"
[DeepSpeech_scorer]: https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer "DeepSpeech_scorer"
[naoqi_bridge_msgs]: https://github.com/ros-naoqi/naoqi_bridge_msgs.git "naoqi_bridge_msgs"
[Numpy]: https://numpy.org "Numpy"
[Plac]: https://pypi.org/project/plac/ "Plac"
[robot_toolkit_msgs]: https://github.com/SinfonIAUniandes/robot_toolkit_msgs.git "robot_toolkit_msgs"
[Scipy]: https://scipy.org/install/ "Scipy"
[soundfile]: https://pypi.org/project/soundfile/ "soundfile"
[Spacy]: https://spacy.io/usage "Spacy"
[SpeechRecognition]: https://pypi.org/project/SpeechRecognition/ "SpeechRecognition"
[speech_utilities_msgs]: https://github.com/SinfonIAUniandes/speech_recognition_msgs.git "speech_utilities_msgs"
[termcolor]: https://pypi.org/project/termcolor/ "termcolor"
[Vosk]: https://pypi.org/project/vosk/ "Vosk"
[Vosk_model]: https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip "Vosk_model"
