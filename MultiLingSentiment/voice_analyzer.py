import streamlit as st
import json
import base64

def create_speech_recognition_component():
    """Create a JavaScript component for speech recognition"""
    
    speech_recognition_js = """
    <div id="speech-container">
        <button id="start-btn" onclick="startRecording()" style="
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        ">🎤 Start Recording</button>
        
        <button id="stop-btn" onclick="stopRecording()" disabled style="
            background-color: #666;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        ">⏹️ Stop Recording</button>
        
        <div id="status" style="margin: 10px 0; color: #666;"></div>
        <div id="transcript" style="
            border: 1px solid #ddd;
            padding: 10px;
            margin: 10px 0;
            min-height: 100px;
            background-color: #f9f9f9;
            border-radius: 5px;
        "></div>
    </div>

    <script>
    let recognition;
    let isRecording = false;
    
    function initSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            recognition = new webkitSpeechRecognition();
        } else if ('SpeechRecognition' in window) {
            recognition = new SpeechRecognition();
        } else {
            document.getElementById('status').innerHTML = 
                '<span style="color: red;">Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.</span>';
            return false;
        }
        
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US'; // Default to English, can be changed
        
        recognition.onstart = function() {
            document.getElementById('status').innerHTML = 
                '<span style="color: green;">🎤 Listening... Speak now!</span>';
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            isRecording = true;
        };
        
        recognition.onresult = function(event) {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            document.getElementById('transcript').innerHTML = 
                '<strong>Final:</strong> ' + finalTranscript + 
                '<br><strong>Interim:</strong> <em>' + interimTranscript + '</em>';
            
            // Send final transcript to Streamlit
            if (finalTranscript) {
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: finalTranscript
                }, '*');
            }
        };
        
        recognition.onerror = function(event) {
            document.getElementById('status').innerHTML = 
                '<span style="color: red;">Error: ' + event.error + '</span>';
            stopRecording();
        };
        
        recognition.onend = function() {
            if (isRecording) {
                document.getElementById('status').innerHTML = 
                    '<span style="color: orange;">Recording stopped</span>';
            }
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            isRecording = false;
        };
        
        return true;
    }
    
    function startRecording() {
        if (!recognition && !initSpeechRecognition()) {
            return;
        }
        
        document.getElementById('transcript').innerHTML = '';
        recognition.start();
    }
    
    function stopRecording() {
        if (recognition && isRecording) {
            recognition.stop();
        }
    }
    
    // Set language for different regions
    function setLanguage(lang) {
        if (recognition) {
            recognition.lang = lang;
        }
    }
    
    // Initialize on load
    window.onload = function() {
        initSpeechRecognition();
    };
    </script>
    """
    
    return speech_recognition_js

def display_voice_recorder():
    """Display the voice recorder component"""
    st.subheader("🎤 Real-time Speech Recognition")
    st.write("Click 'Start Recording' and speak. Your speech will be converted to text automatically.")
    
    # Language selection
    language_options = {
        "English": "en-US",
        "Hindi": "hi-IN", 
        "Telugu": "te-IN"
    }
    
    selected_language = st.selectbox(
        "Select Language for Speech Recognition:",
        options=list(language_options.keys()),
        index=0
    )
    
    # Display the speech recognition component
    speech_component = create_speech_recognition_component()
    
    # Modify the component to use selected language
    speech_component = speech_component.replace(
        "recognition.lang = 'en-US';", 
        f"recognition.lang = '{language_options[selected_language]}';"
    )
    
    st.components.v1.html(speech_component, height=300)
    
    # Manual input fallback
    st.subheader("Manual Input (Fallback)")
    manual_text = st.text_area(
        "If speech recognition doesn't work, type your text here:",
        placeholder="Enter your text manually...",
        height=100
    )
    
    return manual_text

def analyze_voice_input(text, simple_sentiment_analysis, detect_language_simple):
    """Analyze voice input text for sentiment"""
    if not text or not text.strip():
        return None
    
    # Detect language
    detected_language = detect_language_simple(text)
    
    # Analyze sentiment
    result = simple_sentiment_analysis(text)
    
    return {
        'text': text,
        'language': detected_language,
        'sentiment': result
    }