import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sentiment_analyzer import SentimentAnalyzer
from language_detector import LanguageDetector
import time

# Initialize components
@st.cache_resource
def load_sentiment_analyzer():
    """Load and cache the sentiment analyzer"""
    return SentimentAnalyzer()

@st.cache_resource
def load_language_detector():
    """Load and cache the language detector"""
    return LanguageDetector()

def create_confidence_chart(sentiment_scores):
    """Create a bar chart showing confidence scores for each sentiment"""
    labels = list(sentiment_scores.keys())
    scores = [score * 100 for score in sentiment_scores.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker_color=['#ff6b6b' if label == 'NEGATIVE' else '#4ecdc4' if label == 'POSITIVE' else '#ffd93d' for label in labels],
            text=[f'{score:.1f}%' for score in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Confidence Scores",
        xaxis_title="Sentiment",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

def analyze_single_text(text, sentiment_analyzer, language_detector):
    """Analyze sentiment for a single text input"""
    if not text.strip():
        st.warning("Please enter some text to analyze.")
        return
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.spinner('Analyzing sentiment...'):
            # Detect language
            detected_language = language_detector.detect_language(text)
            
            # Analyze sentiment
            result = sentiment_analyzer.analyze_sentiment(text)
            
            if result['error']:
                st.error(f"Error: {result['error']}")
                return
            
            # Display results
            st.subheader("Analysis Results")
            
            # Language detection
            st.info(f"**Detected Language:** {detected_language['language']} ({detected_language['confidence']:.2f} confidence)")
            
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
    
    with col2:
        # Display confidence chart
        if not result['error']:
            fig = create_confidence_chart(result['all_scores'])
            st.plotly_chart(fig, use_container_width=True)

def analyze_batch_text(uploaded_file, sentiment_analyzer, language_detector):
    """Analyze sentiment for batch text from CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if required column exists
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        if not text_columns:
            st.error("No 'text' column found in the CSV file. Please ensure your CSV has a column containing text data.")
            return
        
        text_column = text_columns[0]
        st.info(f"Using column '{text_column}' for analysis")
        
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
            
            # Detect language
            detected_language = language_detector.detect_language(text)
            
            # Analyze sentiment
            sentiment_result = sentiment_analyzer.analyze_sentiment(text)
            
            results.append({
                'Original_Text': text,
                'Detected_Language': detected_language['language'],
                'Language_Confidence': detected_language['confidence'],
                'Sentiment': sentiment_result['sentiment'] if not sentiment_result['error'] else 'ERROR',
                'Sentiment_Confidence': sentiment_result['confidence'] if not sentiment_result['error'] else 0,
                'Positive_Score': sentiment_result['all_scores'].get('POSITIVE', 0) if not sentiment_result['error'] else 0,
                'Neutral_Score': sentiment_result['all_scores'].get('NEUTRAL', 0) if not sentiment_result['error'] else 0,
                'Negative_Score': sentiment_result['all_scores'].get('NEGATIVE', 0) if not sentiment_result['error'] else 0,
                'Error': sentiment_result['error'] if sentiment_result['error'] else None
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            positive_count = len(results_df[results_df['Sentiment'] == 'POSITIVE'])
            st.metric("Positive", positive_count)
        
        with col2:
            neutral_count = len(results_df[results_df['Sentiment'] == 'NEUTRAL'])
            st.metric("Neutral", neutral_count)
        
        with col3:
            negative_count = len(results_df[results_df['Sentiment'] == 'NEGATIVE'])
            st.metric("Negative", negative_count)
        
        with col4:
            error_count = len(results_df[results_df['Sentiment'] == 'ERROR'])
            st.metric("Errors", error_count)
        
        # Create sentiment distribution chart
        if len(results_df) > 0:
            sentiment_counts = results_df['Sentiment'].value_counts()
            if len(sentiment_counts) > 0:
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'POSITIVE': '#4ecdc4',
                        'NEGATIVE': '#ff6b6b',
                        'NEUTRAL': '#ffd93d',
                        'ERROR': '#gray'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
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

def main():
    st.set_page_config(
        page_title="Multi-Language Sentiment Analyzer",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Multi-Language Sentiment Analyzer")
    st.markdown("Analyze sentiment in English, Hindi, and Telugu using advanced AI models")
    
    # Load components
    try:
        sentiment_analyzer = load_sentiment_analyzer()
        language_detector = load_language_detector()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()
    
    # Sidebar for model information
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.write("**Sentiment Model:** xlm-roberta-base")
        st.write("**Language Detection:** langdetect")
        st.write("**Supported Languages:**")
        st.write("- English")
        st.write("- Hindi (हिन्दी)")
        st.write("- Telugu (తెలుగు)")
        
        st.header("📝 Usage Tips")
        st.write("- Enter text in any supported language")
        st.write("- Language will be detected automatically")
        st.write("- Confidence scores show model certainty")
        st.write("- Use batch mode for multiple texts")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Single Text Analysis", "Batch Analysis"])
    
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
            analyze_single_text(text_input, sentiment_analyzer, language_detector)
        
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
                st.session_state.example_text = "ఇది చాలా బాగుంది! నాకు చాలా నచ్చింది."
        
        # Display example text if selected
        if hasattr(st.session_state, 'example_text'):
            st.text_area("Example text:", value=st.session_state.example_text, height=100, disabled=True)
            if st.button("Analyze Example", type="secondary"):
                analyze_single_text(st.session_state.example_text, sentiment_analyzer, language_detector)
    
    with tab2:
        st.header("Batch Analysis")
        st.write("Upload a CSV file with text data for batch sentiment analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should contain a column with text data (column name should contain 'text')"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Preview the file
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.subheader("File Preview")
                st.dataframe(preview_df.head(), use_container_width=True)
                
                # Reset file pointer for processing
                uploaded_file.seek(0)
                
                if st.button("Start Batch Analysis", type="primary"):
                    analyze_batch_text(uploaded_file, sentiment_analyzer, language_detector)
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
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
