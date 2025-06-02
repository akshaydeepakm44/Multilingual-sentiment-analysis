import streamlit as st
import pandas as pd
import base64
import io
from google_speech import transcribe_audio_google, get_language_code_for_speech, test_google_speech_api

# Simple sentiment analysis without external AI libraries
def simple_sentiment_analysis(text):
    """Simple rule-based sentiment analysis"""
    if not text:
        return {"sentiment": "NEUTRAL", "confidence": 0.0}
    
    text = text.lower()
    
    # Positive words
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 'love', 'like', 'happy',
        'best', 'perfect', 'brilliant', 'outstanding', 'superb', 'marvelous', 'terrific', 'fabulous',
        # Hindi positive words (transliterated)
        'accha', 'achha', 'bahut', 'sundar', 'khushi', 'prasanna', 'uttam', 'shandar',
        # Telugu positive words (transliterated)
        'bagundi', 'manchidi', 'santosham', 'bavundi', 'chala', 'bagunna'
    ]
    
    # Negative words
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad', 'angry', 'worst',
        'poor', 'disappointing', 'disgusting', 'pathetic', 'useless', 'annoying', 'frustrating',
        # Hindi negative words (transliterated)
        'bura', 'kharab', 'ganda', 'dukh', 'pareshani', 'galat', 'bekaar',
        # Telugu negative words (transliterated)
        'chedu', 'daridram', 'kastam', 'badha', 'kopam', 'vishaadam'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
        confidence = min(90, 60 + (positive_count - negative_count) * 10)
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
        confidence = min(90, 60 + (negative_count - positive_count) * 10)
    else:
        sentiment = "NEUTRAL"
        confidence = 50
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_score": positive_count,
        "negative_score": negative_count
    }

def detect_language_simple(text):
    """Simple language detection based on character patterns"""
    if not text:
        return {"language": "Unknown", "confidence": 0.0}
    
    # Count different script characters
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')  # Devanagari script
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')  # Telugu script
    
    total_chars = english_chars + hindi_chars + telugu_chars
    
    if total_chars == 0:
        return {"language": "Unknown", "confidence": 0.0}
    
    if hindi_chars > english_chars and hindi_chars > telugu_chars:
        return {"language": "Hindi", "confidence": (hindi_chars / total_chars) * 100}
    elif telugu_chars > english_chars and telugu_chars > hindi_chars:
        return {"language": "Telugu", "confidence": (telugu_chars / total_chars) * 100}
    else:
        return {"language": "English", "confidence": (english_chars / total_chars) * 100}

def main():
    st.set_page_config(
        page_title="Multi-Language Sentiment Analyzer",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Multi-Language Sentiment Analyzer")
    st.markdown("Analyze sentiment in English, Hindi, and Telugu")
    
    # Note about the current version
    st.info("Note: This is a simplified version using rule-based analysis. For AI-powered analysis with HuggingFace models, additional dependencies need to be installed.")
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ Available Features")
        st.write("**Text Analysis:** Rule-based sentiment analysis")
        st.write("**Voice Analysis:** Speech recognition & audio processing")
        st.write("**Video Analysis:** Multi-modal sentiment detection")
        st.write("**Batch Processing:** CSV file analysis")
        st.write("**Language Detection:** Script-based identification")
        
        st.header("🌐 Supported Languages")
        st.write("- English")
        st.write("- Hindi (हिन्दी)")
        st.write("- Telugu (తెలుగు)")
        
        st.header("📝 Quick Guide")
        st.write("- **Text:** Direct input analysis")
        st.write("- **Voice:** Browser speech recognition")
        st.write("- **Video:** Upload files for analysis")
        st.write("- **Batch:** CSV upload for multiple texts")
        
        st.header("🔧 Enhancement Options")
        if st.button("Enable AI Models", key="enable_ai"):
            st.info("For AI-powered analysis using HuggingFace transformers, additional dependencies are needed. This would provide more accurate sentiment detection.")
        
        if st.button("Setup Video Processing", key="enable_video"):
            st.info("For automatic video processing (facial recognition, audio extraction), computer vision libraries are required.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Single Text Analysis", "Voice Analysis", "Video Analysis", "Batch Analysis"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        text_input = st.text_area(
            "Enter text for sentiment analysis:",
            height=150,
            placeholder="Type your text here in English, Hindi, or Telugu..."
        )
        
        # Analysis button
        if st.button("Analyze Sentiment", type="primary"):
            if text_input.strip():
                # Create columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.spinner('Analyzing sentiment...'):
                        # Detect language
                        detected_language = detect_language_simple(text_input)
                        
                        # Analyze sentiment
                        result = simple_sentiment_analysis(text_input)
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Language detection
                        st.info(f"**Detected Language:** {detected_language['language']} ({detected_language['confidence']:.1f}% confidence)")
                        
                        # Main sentiment result
                        sentiment = result['sentiment']
                        confidence = result['confidence']
                        
                        # Color code the sentiment
                        if sentiment == 'POSITIVE':
                            st.success(f"**Sentiment:** {sentiment} ({confidence:.1f}% confidence)")
                        elif sentiment == 'NEGATIVE':
                            st.error(f"**Sentiment:** {sentiment} ({confidence:.1f}% confidence)")
                        else:
                            st.warning(f"**Sentiment:** {sentiment} ({confidence:.1f}% confidence)")
                        
                        # Show word counts
                        st.write(f"Positive indicators: {result['positive_score']}")
                        st.write(f"Negative indicators: {result['negative_score']}")
                
                with col2:
                    # Simple confidence display
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                    st.metric("Positive Words", result['positive_score'])
                    st.metric("Negative Words", result['negative_score'])
            else:
                st.warning("Please enter some text to analyze.")
        
        # Example texts
        st.subheader("Try These Examples:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("English Example"):
                st.session_state.example_text = "I love this new technology! It's absolutely amazing and works perfectly."
        
        with col2:
            if st.button("Hindi Example"):
                st.session_state.example_text = "यह बहुत अच्छा है! मुझे यह बहुत पसंद आया।"
        
        with col3:
            if st.button("Telugu Example"):
                st.session_state.example_text = "ఇది చాలా బాగుంది! నాకు చాలా నచ్చింది।"
        
        # Display example text if selected
        if hasattr(st.session_state, 'example_text'):
            st.text_area("Example text:", value=st.session_state.example_text, height=100, disabled=True)
            if st.button("Analyze Example", type="secondary"):
                # Analyze the example text
                detected_language = detect_language_simple(st.session_state.example_text)
                result = simple_sentiment_analysis(st.session_state.example_text)
                
                st.subheader("Example Analysis Results")
                st.info(f"**Language:** {detected_language['language']} ({detected_language['confidence']:.1f}% confidence)")
                
                if result['sentiment'] == 'POSITIVE':
                    st.success(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                elif result['sentiment'] == 'NEGATIVE':
                    st.error(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                else:
                    st.warning(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
    
    with tab2:
        st.header("🎤 Voice Analysis")
        st.write("Record your voice or upload an audio file for sentiment analysis")
        
        # Note about voice analysis
        st.info("Voice analysis converts speech to text first, then analyzes sentiment. Supports English, Hindi, and Telugu.")
        
        # Two options for voice input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📹 Record Audio")
            
            # Test Google Speech API availability
            api_available, api_status = test_google_speech_api()
            
            if api_available:
                st.success("Google Speech-to-Text API is ready for automatic transcription")
            else:
                st.warning(f"Google Speech API status: {api_status}")
            
            # Language selection for speech recognition
            speech_language = st.selectbox(
                "Select speech language:",
                ["English", "Hindi", "Telugu"],
                key="speech_language_selection"
            )
            
            # Audio input widget
            audio_bytes = st.audio_input("Record your voice")
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                col_auto, col_manual = st.columns(2)
                
                with col_auto:
                    st.write("**Automatic Transcription:**")
                    if st.button("Auto-Transcribe with Google API", type="primary") and api_available:
                        with st.spinner("Transcribing audio..."):
                            language_code = get_language_code_for_speech(speech_language)
                            transcription_result = transcribe_audio_google(audio_bytes.getvalue(), language_code)
                            
                            if transcription_result.get("error"):
                                st.error(f"Transcription failed: {transcription_result['error']}")
                                st.info("Please use manual transcription below")
                            else:
                                transcribed_text = transcription_result["text"]
                                confidence = transcription_result.get("confidence", 0) * 100
                                
                                st.success(f"Transcription successful (Confidence: {confidence:.1f}%)")
                                st.text_area("Transcribed text:", value=transcribed_text, height=80, disabled=True)
                                
                                # Auto-analyze the transcribed text
                                detected_language = detect_language_simple(transcribed_text)
                                result = simple_sentiment_analysis(transcribed_text)
                                
                                st.subheader("Voice Analysis Results")
                                st.info(f"**Detected Language:** {detected_language['language']} ({detected_language['confidence']:.1f}% confidence)")
                                
                                if result['sentiment'] == 'POSITIVE':
                                    st.success(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                                elif result['sentiment'] == 'NEGATIVE':
                                    st.error(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                                else:
                                    st.warning(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                                
                                # Additional metrics
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                                with col_b:
                                    st.metric("Positive Words", result['positive_score'])
                                with col_c:
                                    st.metric("Negative Words", result['negative_score'])
                
                with col_manual:
                    st.write("**Manual Transcription:**")
                    manual_transcribed_text = st.text_area(
                        "Type what you said:",
                        height=100,
                        placeholder="Enter the text from your voice recording...",
                        key="manual_transcription"
                    )
                    
                    if st.button("Analyze Manual Transcription", type="secondary") and manual_transcribed_text:
                        # Analyze the manually transcribed text
                        detected_language = detect_language_simple(manual_transcribed_text)
                        result = simple_sentiment_analysis(manual_transcribed_text)
                        
                        st.subheader("Manual Analysis Results")
                        st.info(f"**Detected Language:** {detected_language['language']} ({detected_language['confidence']:.1f}% confidence)")
                        
                        if result['sentiment'] == 'POSITIVE':
                            st.success(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                        elif result['sentiment'] == 'NEGATIVE':
                            st.error(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                        else:
                            st.warning(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                        
                        # Show word count
                        word_count = len(manual_transcribed_text.split())
                        st.info(f"**Analysis:** {word_count} words processed")
        
        with col2:
            st.subheader("🔊 Advanced Speech Recognition")
            
            # Web Speech API component
            st.write("**Browser-based Speech Recognition:**")
            
            # Language selection for speech recognition
            speech_lang_options = {
                "English": "en-US",
                "Hindi": "hi-IN", 
                "Telugu": "te-IN"
            }
            
            selected_speech_lang = st.selectbox(
                "Speech Recognition Language:",
                options=list(speech_lang_options.keys()),
                key="speech_lang"
            )
            
            # HTML component for speech recognition
            speech_recognition_html = f"""
            <div style="border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;">
                <h4>🎤 Real-time Speech Recognition</h4>
                
                <button id="start-btn" onclick="startRecording()" style="
                    background-color: #ff4b4b;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 5px;
                    font-size: 14px;
                ">🎤 Start Recording</button>
                
                <button id="stop-btn" onclick="stopRecording()" disabled style="
                    background-color: #666;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 5px;
                    font-size: 14px;
                ">⏹️ Stop Recording</button>
                
                <div id="status" style="margin: 10px 0; color: #666; font-weight: bold;"></div>
                <div id="transcript" style="
                    border: 1px solid #ccc;
                    padding: 15px;
                    margin: 10px 0;
                    min-height: 80px;
                    background-color: white;
                    border-radius: 5px;
                    font-family: 'Courier New', monospace;
                ">Recognized text will appear here...</div>
                
                <button onclick="copyTranscript()" style="
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 3px;
                    cursor: pointer;
                    font-size: 12px;
                ">📋 Copy Text</button>
            </div>

            <script>
            let recognition;
            let isRecording = false;
            let finalTranscript = '';
            
            function initSpeechRecognition() {{
                if ('webkitSpeechRecognition' in window) {{
                    recognition = new webkitSpeechRecognition();
                }} else if ('SpeechRecognition' in window) {{
                    recognition = new SpeechRecognition();
                }} else {{
                    document.getElementById('status').innerHTML = 
                        '<span style="color: red;">Speech recognition not supported. Please use Chrome, Edge, or Safari.</span>';
                    document.getElementById('start-btn').disabled = true;
                    return false;
                }}
                
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = '{speech_lang_options[selected_speech_lang]}';
                
                recognition.onstart = function() {{
                    document.getElementById('status').innerHTML = 
                        '<span style="color: green;">🎤 Listening in {selected_speech_lang}... Speak now!</span>';
                    document.getElementById('start-btn').disabled = true;
                    document.getElementById('stop-btn').disabled = false;
                    isRecording = true;
                }};
                
                recognition.onresult = function(event) {{
                    let interimTranscript = '';
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {{
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {{
                            finalTranscript += transcript + ' ';
                        }} else {{
                            interimTranscript += transcript;
                        }}
                    }}
                    
                    document.getElementById('transcript').innerHTML = 
                        '<strong>Final:</strong> ' + finalTranscript + 
                        '<br><strong>Speaking:</strong> <em style="color: #666;">' + interimTranscript + '</em>';
                }};
                
                recognition.onerror = function(event) {{
                    document.getElementById('status').innerHTML = 
                        '<span style="color: red;">Error: ' + event.error + '</span>';
                    stopRecording();
                }};
                
                recognition.onend = function() {{
                    document.getElementById('status').innerHTML = 
                        '<span style="color: orange;">Recording stopped. Final transcript ready.</span>';
                    document.getElementById('start-btn').disabled = false;
                    document.getElementById('stop-btn').disabled = true;
                    isRecording = false;
                }};
                
                return true;
            }}
            
            function startRecording() {{
                if (!recognition && !initSpeechRecognition()) {{
                    return;
                }}
                
                finalTranscript = '';
                document.getElementById('transcript').innerHTML = 'Listening...';
                recognition.start();
            }}
            
            function stopRecording() {{
                if (recognition && isRecording) {{
                    recognition.stop();
                }}
            }}
            
            function copyTranscript() {{
                const text = finalTranscript.trim();
                if (text) {{
                    navigator.clipboard.writeText(text).then(function() {{
                        alert('Text copied to clipboard!');
                    }});
                }} else {{
                    alert('No text to copy');
                }}
            }}
            
            // Initialize on load
            window.onload = function() {{
                initSpeechRecognition();
            }};
            </script>
            """
            
            st.markdown(speech_recognition_html, unsafe_allow_html=True)
            
            # Manual input for recognized text
            st.write("**Copy recognized text here for analysis:**")
            voice_text_input = st.text_area(
                "Paste or type the recognized text:",
                height=100,
                placeholder="Paste the recognized text from above or type manually...",
                key="voice_text_analysis"
            )
            
            if st.button("Analyze Voice Text", type="primary") and voice_text_input:
                # Analyze the voice text
                detected_language = detect_language_simple(voice_text_input)
                result = simple_sentiment_analysis(voice_text_input)
                
                st.subheader("Voice Analysis Results")
                st.info(f"**Detected Language:** {detected_language['language']} ({detected_language['confidence']:.1f}% confidence)")
                
                if result['sentiment'] == 'POSITIVE':
                    st.success(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                elif result['sentiment'] == 'NEGATIVE':
                    st.error(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                else:
                    st.warning(f"**Sentiment:** {result['sentiment']} ({result['confidence']:.1f}% confidence)")
                
                # Voice-specific metrics
                voice_col1, voice_col2, voice_col3 = st.columns(3)
                with voice_col1:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                with voice_col2:
                    st.metric("Positive Words", result['positive_score'])
                with voice_col3:
                    st.metric("Negative Words", result['negative_score'])
                    
                # Show word count and speaking rate estimate
                word_count = len(voice_text_input.split())
                st.info(f"**Speech Analysis:** {word_count} words detected")
        
        # Voice analysis tips
        st.subheader("🎯 Voice Analysis Tips")
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        
        with tip_col1:
            st.write("**Recording Quality**")
            st.write("- Speak clearly and slowly")
            st.write("- Minimize background noise")
            st.write("- Use a good microphone if available")
        
        with tip_col2:
            st.write("**Language Support**")
            st.write("- English: Full support")
            st.write("- Hindi: Basic support")
            st.write("- Telugu: Basic support")
        
        with tip_col3:
            st.write("**Best Practices**")
            st.write("- Keep recordings under 1 minute")
            st.write("- Speak one language at a time")
            st.write("- Review transcription for accuracy")
        
        # Automatic speech recognition setup
        st.subheader("🔧 Automatic Speech Recognition Setup")
        st.write("For automatic speech-to-text conversion, you can integrate with speech recognition services:")
        
        setup_col1, setup_col2 = st.columns(2)
        with setup_col1:
            st.write("**Google Speech-to-Text**")
            st.write("- Supports 125+ languages")
            st.write("- High accuracy for multilingual content")
            st.write("- Requires Google Cloud API key")
        
        with setup_col2:
            st.write("**Web Speech API**")
            st.write("- Browser-based recognition")
            st.write("- No API key required")
            st.write("- Limited language support")
        
        if st.button("Enable Automatic Speech Recognition"):
            st.info("To enable automatic speech recognition, please provide your Google Speech-to-Text API credentials. This will allow the app to automatically convert speech to text without manual transcription.")
    
    with tab3:
        st.header("🎬 Video Sentiment Analysis")
        st.write("Analyze sentiment from uploaded videos using facial expressions and audio")
        
        # Note about video analysis
        st.info("Video analysis extracts audio for speech sentiment and analyzes facial expressions for emotion detection. This feature provides multi-modal sentiment analysis.")
        
        # Video upload section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Upload Video")
            
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload video file in MP4, AVI, MOV, MKV, or WebM format"
            )
            
            if uploaded_video:
                st.video(uploaded_video)
                st.success(f"Video uploaded: {uploaded_video.name}")
                
                # Video information
                file_size = len(uploaded_video.getvalue()) / (1024 * 1024)  # MB
                st.info(f"File size: {file_size:.2f} MB")
                
                # Analysis options
                st.subheader("Analysis Options")
                
                analysis_type = st.radio(
                    "Select analysis type:",
                    ["Audio Only (Speech Sentiment)", "Visual Only (Facial Expression)", "Combined Analysis"],
                    index=2
                )
                
                # Time range selection
                st.write("**Analysis Duration:**")
                duration_option = st.selectbox(
                    "Select duration to analyze:",
                    ["First 30 seconds", "First 1 minute", "Entire video", "Custom range"]
                )
                
                if duration_option == "Custom range":
                    time_col1, time_col2 = st.columns(2)
                    with time_col1:
                        start_time = st.number_input("Start time (seconds)", min_value=0, value=0)
                    with time_col2:
                        end_time = st.number_input("End time (seconds)", min_value=1, value=30)
                
                if st.button("Analyze Video", type="primary"):
                    with st.spinner("Processing video... This may take a moment."):
                        
                        # Audio Analysis Section
                        if analysis_type in ["Audio Only (Speech Sentiment)", "Combined Analysis"]:
                            st.subheader("🔊 Audio Analysis Results")
                            
                            # For now, we'll ask for manual transcription
                            st.warning("Audio extraction from video requires additional setup. Please provide the transcript of the spoken content below.")
                            
                            video_transcript = st.text_area(
                                "Transcribe the audio from your video:",
                                height=120,
                                placeholder="Enter what was spoken in the video...",
                                key="video_transcript"
                            )
                            
                            if video_transcript:
                                # Analyze the transcribed audio
                                audio_language = detect_language_simple(video_transcript)
                                audio_sentiment = simple_sentiment_analysis(video_transcript)
                                
                                st.info(f"**Audio Language:** {audio_language['language']} ({audio_language['confidence']:.1f}% confidence)")
                                
                                if audio_sentiment['sentiment'] == 'POSITIVE':
                                    st.success(f"**Audio Sentiment:** {audio_sentiment['sentiment']} ({audio_sentiment['confidence']:.1f}% confidence)")
                                elif audio_sentiment['sentiment'] == 'NEGATIVE':
                                    st.error(f"**Audio Sentiment:** {audio_sentiment['sentiment']} ({audio_sentiment['confidence']:.1f}% confidence)")
                                else:
                                    st.warning(f"**Audio Sentiment:** {audio_sentiment['sentiment']} ({audio_sentiment['confidence']:.1f}% confidence)")
                        
                        # Visual Analysis Section
                        if analysis_type in ["Visual Only (Facial Expression)", "Combined Analysis"]:
                            st.subheader("😊 Facial Expression Analysis")
                            
                            # Simulated facial expression analysis
                            st.warning("Facial expression analysis requires computer vision libraries. Please describe the general facial expressions observed in the video.")
                            
                            facial_expressions = st.multiselect(
                                "Select observed facial expressions:",
                                ["Happy/Smiling", "Sad/Frowning", "Neutral", "Surprised", "Angry", "Fearful", "Disgusted"],
                                default=["Neutral"]
                            )
                            
                            if facial_expressions:
                                # Simple mapping of expressions to sentiment
                                positive_expressions = ["Happy/Smiling", "Surprised"]
                                negative_expressions = ["Sad/Frowning", "Angry", "Fearful", "Disgusted"]
                                
                                positive_count = sum(1 for expr in facial_expressions if expr in positive_expressions)
                                negative_count = sum(1 for expr in facial_expressions if expr in negative_expressions)
                                
                                if positive_count > negative_count:
                                    facial_sentiment = "POSITIVE"
                                    facial_confidence = min(85, 60 + positive_count * 15)
                                elif negative_count > positive_count:
                                    facial_sentiment = "NEGATIVE"
                                    facial_confidence = min(85, 60 + negative_count * 15)
                                else:
                                    facial_sentiment = "NEUTRAL"
                                    facial_confidence = 70
                                
                                if facial_sentiment == 'POSITIVE':
                                    st.success(f"**Facial Sentiment:** {facial_sentiment} ({facial_confidence:.1f}% confidence)")
                                elif facial_sentiment == 'NEGATIVE':
                                    st.error(f"**Facial Sentiment:** {facial_sentiment} ({facial_confidence:.1f}% confidence)")
                                else:
                                    st.warning(f"**Facial Sentiment:** {facial_sentiment} ({facial_confidence:.1f}% confidence)")
                        
                        # Combined Analysis Results
                        if analysis_type == "Combined Analysis" and video_transcript and facial_expressions:
                            st.subheader("🎯 Combined Analysis Results")
                            
                            # Weighted combination of audio and visual sentiment
                            audio_weight = 0.6  # Give more weight to audio
                            visual_weight = 0.4
                            
                            # Convert sentiments to numerical scores
                            sentiment_scores = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
                            
                            audio_score = sentiment_scores[audio_sentiment['sentiment']]
                            visual_score = sentiment_scores[facial_sentiment]
                            
                            combined_score = (audio_score * audio_weight) + (visual_score * visual_weight)
                            
                            if combined_score > 0.3:
                                combined_sentiment = "POSITIVE"
                                combined_confidence = min(90, 70 + abs(combined_score) * 20)
                            elif combined_score < -0.3:
                                combined_sentiment = "NEGATIVE"
                                combined_confidence = min(90, 70 + abs(combined_score) * 20)
                            else:
                                combined_sentiment = "NEUTRAL"
                                combined_confidence = 65
                            
                            if combined_sentiment == 'POSITIVE':
                                st.success(f"**Overall Video Sentiment:** {combined_sentiment} ({combined_confidence:.1f}% confidence)")
                            elif combined_sentiment == 'NEGATIVE':
                                st.error(f"**Overall Video Sentiment:** {combined_sentiment} ({combined_confidence:.1f}% confidence)")
                            else:
                                st.warning(f"**Overall Video Sentiment:** {combined_sentiment} ({combined_confidence:.1f}% confidence)")
                            
                            # Detailed breakdown
                            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
                            with breakdown_col1:
                                st.metric("Audio Weight", f"{audio_weight*100:.0f}%")
                            with breakdown_col2:
                                st.metric("Visual Weight", f"{visual_weight*100:.0f}%")
                            with breakdown_col3:
                                st.metric("Combined Score", f"{combined_score:.2f}")
        
        with col2:
            st.subheader("🛠️ Video Analysis Setup")
            
            # Technical requirements
            st.write("**Required Libraries for Full Video Analysis:**")
            
            setup_info = {
                "OpenCV": "For video frame extraction and processing",
                "MoviePy": "For audio extraction from video files", 
                "DeepFace": "For facial emotion recognition",
                "FER": "Alternative facial expression recognition",
                "Speech Recognition": "For automatic audio transcription"
            }
            
            for lib, description in setup_info.items():
                st.write(f"• **{lib}:** {description}")
            
            st.write("**Current Status:**")
            st.warning("Video processing libraries not installed. Manual analysis available.")
            
            # Setup button
            if st.button("Install Video Analysis Libraries"):
                st.info("Video analysis requires additional dependencies. These libraries enable automatic facial expression detection and audio extraction from video files.")
            
            # Analysis tips
            st.subheader("📋 Video Analysis Tips")
            
            tips = [
                "Keep videos under 2 minutes for faster processing",
                "Ensure good lighting for facial expression analysis",
                "Clear audio improves speech sentiment accuracy",
                "Multiple faces may affect emotion detection",
                "Combine audio and visual for best results"
            ]
            
            for tip in tips:
                st.write(f"• {tip}")
            
            # Sample results format
            st.subheader("📊 Sample Analysis Output")
            
            sample_results = {
                "Audio Sentiment": "POSITIVE (78%)",
                "Facial Expression": "Happy/Neutral (85%)",
                "Combined Result": "POSITIVE (81%)",
                "Confidence": "High"
            }
            
            for metric, value in sample_results.items():
                st.write(f"**{metric}:** {value}")
    
    with tab4:
        st.header("Batch Analysis")
        st.write("Upload a CSV file with text data for batch sentiment analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should contain a column with text data (column name should contain 'text')"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Show file preview
                st.subheader("File Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for text column
                text_columns = [col for col in df.columns if 'text' in col.lower()]
                if not text_columns:
                    st.error("No 'text' column found in the CSV file. Please ensure your CSV has a column containing text data.")
                else:
                    text_column = text_columns[0]
                    st.info(f"Using column '{text_column}' for analysis")
                    
                    if st.button("Start Batch Analysis", type="primary"):
                        # Initialize results
                        results = []
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_rows = len(df)
                        
                        for idx, row in df.iterrows():
                            text = str(row[text_column])
                            
                            # Update progress
                            progress = (idx + 1) / total_rows
                            progress_bar.progress(progress)
                            status_text.text(f'Analyzing text {idx + 1} of {total_rows}')
                            
                            # Detect language and analyze sentiment
                            detected_language = detect_language_simple(text)
                            sentiment_result = simple_sentiment_analysis(text)
                            
                            results.append({
                                'Original_Text': text,
                                'Detected_Language': detected_language['language'],
                                'Language_Confidence': detected_language['confidence'],
                                'Sentiment': sentiment_result['sentiment'],
                                'Sentiment_Confidence': sentiment_result['confidence'],
                                'Positive_Words': sentiment_result['positive_score'],
                                'Negative_Words': sentiment_result['negative_score']
                            })
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("Batch Analysis Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Show summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            positive_count = len(results_df[results_df['Sentiment'] == 'POSITIVE'])
                            st.metric("Positive", positive_count)
                        
                        with col2:
                            neutral_count = len(results_df[results_df['Sentiment'] == 'NEUTRAL'])
                            st.metric("Neutral", neutral_count)
                        
                        with col3:
                            negative_count = len(results_df[results_df['Sentiment'] == 'NEGATIVE'])
                            st.metric("Negative", negative_count)
                        
                        # Download button for results
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
        
        # Sample CSV format
        st.subheader("Sample CSV Format")
        sample_data = pd.DataFrame({
            'text': [
                'I love this product!',
                'यह बहुत अच्छा है',
                'ఇది చాలా బాగుంది'
            ]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        # Download sample CSV
        sample_csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv,
            file_name="sample_sentiment_analysis.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()