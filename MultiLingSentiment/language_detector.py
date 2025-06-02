from langdetect import detect, detect_langs, LangDetectError
import logging

# Suppress langdetect warnings
logging.getLogger('langdetect').setLevel(logging.WARNING)

class LanguageDetector:
    """
    Language detection utility for multi-language text processing
    """
    
    def __init__(self):
        """Initialize the language detector"""
        # Mapping of langdetect codes to human-readable names
        self.language_names = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu',
            'ta': 'Tamil',
            'kn': 'Kannada',
            'ml': 'Malayalam',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'mr': 'Marathi',
            'pa': 'Punjabi',
            'or': 'Odia',
            'as': 'Assamese',
            'ur': 'Urdu',
            'ne': 'Nepali',
            'si': 'Sinhala',
            'my': 'Myanmar',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'tl': 'Filipino',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'fa': 'Persian',
            'he': 'Hebrew',
            'tr': 'Turkish',
            'ru': 'Russian',
            'uk': 'Ukrainian',
            'bg': 'Bulgarian',
            'hr': 'Croatian',
            'cs': 'Czech',
            'da': 'Danish',
            'nl': 'Dutch',
            'et': 'Estonian',
            'fi': 'Finnish',
            'fr': 'French',
            'de': 'German',
            'el': 'Greek',
            'hu': 'Hungarian',
            'is': 'Icelandic',
            'ga': 'Irish',
            'it': 'Italian',
            'lv': 'Latvian',
            'lt': 'Lithuanian',
            'mk': 'Macedonian',
            'mt': 'Maltese',
            'no': 'Norwegian',
            'pl': 'Polish',
            'pt': 'Portuguese',
            'ro': 'Romanian',
            'sk': 'Slovak',
            'sl': 'Slovenian',
            'es': 'Spanish',
            'sv': 'Swedish',
            'cy': 'Welsh'
        }
        
        # Supported languages for sentiment analysis
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'te': 'Telugu'
        }
        
        print("✅ Language detector initialized successfully")
    
    def detect_language(self, text):
        """
        Detect the language of the given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing language detection results
        """
        if not text or not text.strip():
            return {
                'language': 'Unknown',
                'language_code': 'unknown',
                'confidence': 0.0,
                'is_supported': False,
                'error': 'Empty text provided'
            }
        
        try:
            # Clean text
            text = text.strip()
            
            # Detect language with confidence scores
            language_probabilities = detect_langs(text)
            
            if not language_probabilities:
                return {
                    'language': 'Unknown',
                    'language_code': 'unknown',
                    'confidence': 0.0,
                    'is_supported': False,
                    'error': 'Could not detect language'
                }
            
            # Get the most probable language
            top_language = language_probabilities[0]
            language_code = top_language.lang
            confidence = top_language.prob
            
            # Get human-readable language name
            language_name = self.language_names.get(language_code, f'Unknown ({language_code})')
            
            # Check if language is supported for sentiment analysis
            is_supported = language_code in self.supported_languages
            
            return {
                'language': language_name,
                'language_code': language_code,
                'confidence': confidence,
                'is_supported': is_supported,
                'all_detections': [
                    {
                        'language': self.language_names.get(lang.lang, f'Unknown ({lang.lang})'),
                        'language_code': lang.lang,
                        'confidence': lang.prob
                    }
                    for lang in language_probabilities[:3]  # Top 3 detections
                ],
                'error': None
            }
            
        except LangDetectError as e:
            error_message = f"Language detection error: {str(e)}"
            print(f"❌ {error_message}")
            return {
                'language': 'Unknown',
                'language_code': 'unknown',
                'confidence': 0.0,
                'is_supported': False,
                'error': error_message
            }
        except Exception as e:
            error_message = f"Unexpected error in language detection: {str(e)}"
            print(f"❌ {error_message}")
            return {
                'language': 'Unknown',
                'language_code': 'unknown',
                'confidence': 0.0,
                'is_supported': False,
                'error': error_message
            }
    
    def is_supported_language(self, text):
        """
        Check if the detected language is supported for sentiment analysis
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if language is supported, False otherwise
        """
        detection_result = self.detect_language(text)
        return detection_result['is_supported']
    
    def get_supported_languages(self):
        """
        Get list of supported languages for sentiment analysis
        
        Returns:
            dict: Dictionary of supported language codes and names
        """
        return self.supported_languages.copy()
    
    def detect_batch(self, texts):
        """
        Detect languages for a batch of texts
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of language detection results
        """
        results = []
        for text in texts:
            result = self.detect_language(text)
            results.append(result)
        return results
    
    def get_language_statistics(self, texts):
        """
        Get language distribution statistics for a batch of texts
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            dict: Language distribution statistics
        """
        language_counts = {}
        total_texts = len(texts)
        
        for text in texts:
            detection = self.detect_language(text)
            language = detection['language']
            language_counts[language] = language_counts.get(language, 0) + 1
        
        # Calculate percentages
        language_percentages = {
            lang: (count / total_texts) * 100
            for lang, count in language_counts.items()
        }
        
        return {
            'total_texts': total_texts,
            'language_counts': language_counts,
            'language_percentages': language_percentages,
            'unique_languages': len(language_counts)
        }
