import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging

# Configure logging
logging.getLogger("transformers").setLevel(logging.WARNING)

class SentimentAnalyzer:
    """
    Multi-language sentiment analyzer using HuggingFace transformers
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_name (str): HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model and tokenizer"""
        try:
            # Load the sentiment analysis pipeline
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            print(f"✅ Sentiment analyzer loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading sentiment analyzer: {str(e)}")
            # Fallback to a more basic model
            try:
                self.classifier = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                print("✅ Fallback sentiment analyzer loaded successfully")
            except Exception as fallback_error:
                print(f"❌ Error loading fallback model: {str(fallback_error)}")
                raise fallback_error
    
    def _normalize_sentiment_labels(self, results):
        """
        Normalize sentiment labels to standard format (POSITIVE, NEGATIVE, NEUTRAL)
        
        Args:
            results (list): Raw results from the model
            
        Returns:
            dict: Normalized sentiment scores
        """
        normalized_scores = {}
        
        for result in results:
            label = result['label'].upper()
            score = result['score']
            
            # Map various label formats to standard ones
            if label in ['POSITIVE', 'POS', 'LABEL_2', '5 STARS', '4 STARS']:
                normalized_scores['POSITIVE'] = normalized_scores.get('POSITIVE', 0) + score
            elif label in ['NEGATIVE', 'NEG', 'LABEL_0', '1 STAR', '2 STARS']:
                normalized_scores['NEGATIVE'] = normalized_scores.get('NEGATIVE', 0) + score
            elif label in ['NEUTRAL', 'NEU', 'LABEL_1', '3 STARS']:
                normalized_scores['NEUTRAL'] = normalized_scores.get('NEUTRAL', 0) + score
            else:
                # For unknown labels, try to infer from the label name
                if 'positive' in label.lower() or 'pos' in label.lower():
                    normalized_scores['POSITIVE'] = normalized_scores.get('POSITIVE', 0) + score
                elif 'negative' in label.lower() or 'neg' in label.lower():
                    normalized_scores['NEGATIVE'] = normalized_scores.get('NEGATIVE', 0) + score
                else:
                    normalized_scores['NEUTRAL'] = normalized_scores.get('NEUTRAL', 0) + score
        
        # Ensure all three categories exist
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if sentiment not in normalized_scores:
                normalized_scores[sentiment] = 0.0
        
        return normalized_scores
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of the given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        if not text or not text.strip():
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'all_scores': {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 1.0},
                'error': 'Empty text provided'
            }
        
        try:
            # Clean and prepare text
            text = text.strip()
            if len(text) > 512:  # Truncate very long texts
                text = text[:512]
            
            # Get predictions
            results = self.classifier(text)
            
            # Handle different output formats
            if isinstance(results[0], list):
                results = results[0]  # Unwrap if nested
            
            # Normalize sentiment labels
            normalized_scores = self._normalize_sentiment_labels(results)
            
            # Find the sentiment with highest confidence
            max_sentiment = max(normalized_scores.items(), key=lambda x: x[1])
            predicted_sentiment = max_sentiment[0]
            confidence = max_sentiment[1] * 100  # Convert to percentage
            
            return {
                'sentiment': predicted_sentiment,
                'confidence': confidence,
                'all_scores': normalized_scores,
                'error': None
            }
            
        except Exception as e:
            error_message = f"Error analyzing sentiment: {str(e)}"
            print(f"❌ {error_message}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'all_scores': {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 1.0},
                'error': error_message
            }
    
    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.classifier is not None
        }
