import os
import base64
import requests
import json
import streamlit as st

def transcribe_audio_google(audio_data, language_code="en-US"):
    """
    Transcribe audio using Google Cloud Speech-to-Text API
    
    Args:
        audio_data: Audio data in bytes
        language_code: Language code (e.g., 'en-US', 'hi-IN', 'te-IN')
    
    Returns:
        dict: Transcription result with text and confidence
    """
    try:
        api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not api_key:
            return {"error": "Google Cloud API key not found"}
        
        # Encode audio data to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Google Speech-to-Text API endpoint
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"
        
        # Request payload
        payload = {
            "config": {
                "encoding": "WEBM_OPUS",  # Common web audio format
                "sampleRateHertz": 48000,
                "languageCode": language_code,
                "alternativeLanguageCodes": ["en-US", "hi-IN", "te-IN"],  # Support multiple languages
                "enableAutomaticPunctuation": True,
                "enableWordTimeOffsets": False
            },
            "audio": {
                "content": audio_base64
            }
        }
        
        # Make API request
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'results' in result and len(result['results']) > 0:
                # Get the best transcription
                best_result = result['results'][0]
                if 'alternatives' in best_result and len(best_result['alternatives']) > 0:
                    transcript = best_result['alternatives'][0]['transcript']
                    confidence = best_result['alternatives'][0].get('confidence', 0.0)
                    
                    return {
                        "text": transcript,
                        "confidence": confidence,
                        "language_detected": language_code,
                        "error": None
                    }
                else:
                    return {"error": "No speech detected in audio"}
            else:
                return {"error": "No transcription results found"}
        else:
            error_msg = f"API request failed: {response.status_code}"
            if response.text:
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', {}).get('message', 'Unknown error')}"
                except:
                    error_msg += f" - {response.text}"
            return {"error": error_msg}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - audio file may be too large"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_language_code_for_speech(language_name):
    """Convert language name to Google Speech API language code"""
    language_mapping = {
        "English": "en-US",
        "Hindi": "hi-IN",
        "Telugu": "te-IN"
    }
    return language_mapping.get(language_name, "en-US")

def test_google_speech_api():
    """Test if Google Speech API is accessible"""
    try:
        api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not api_key:
            return False, "API key not found"
        
        # Test with a minimal request
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"
        
        # Create a minimal test payload
        test_payload = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US"
            },
            "audio": {
                "content": ""  # Empty content for testing API access
            }
        }
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_payload),
            timeout=10
        )
        
        # We expect this to fail due to empty audio, but it should give us a proper API response
        if response.status_code in [200, 400]:  # 400 is expected for empty audio
            return True, "API accessible"
        else:
            return False, f"API not accessible: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"