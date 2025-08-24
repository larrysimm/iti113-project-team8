import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_components():
    try:
        # Check if model file exists
        if not os.path.exists('model.h5'):
            st.error("Model file (model.h5) not found!")
            return None, None
            
        model = load_model('model.h5')
        
        # Try to load .pkl tokenizer file
        if os.path.exists('tokenizer.pkl'):
            with open('tokenizer.pkl', 'rb') as handle:
                tokenizer = pickle.load(handle)
            st.success("Successfully loaded tokenizer.pkl")
        # Try alternative file names if needed
        elif os.path.exists('tokenizer.pickle'):
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            st.success("Successfully loaded tokenizer.pickle")
        else:
            st.error("Tokenizer file not found. Please ensure tokenizer.pkl exists.")
            return model, None
            
        return model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Prediction function
def predict_next_words(text, num_words=3, temperature=1.0):
    if not text.strip():
        return "Please enter some text first"
    
    try:
        # Tokenize input text
        sequence = tokenizer.texts_to_sequences([text])
        
        if not sequence or not sequence[0]:
            return "No recognizable words in input"
            
        sequence = sequence[0]
        
        # Predict next words
        predictions = []
        for _ in range(num_words):
            # Pad sequence
            padded_sequence = pad_sequences([sequence], maxlen=model.input_shape[1], padding='pre')
            
            # Predict
            predicted_probs = model.predict(padded_sequence, verbose=0)[0]
            
            # Apply temperature
            predicted_probs = np.log(predicted_probs) / temperature
            exp_preds = np.exp(predicted_probs)
            predicted_probs = exp_preds / np.sum(exp_preds)
            
            # Sample from distribution
            predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            
            # Convert index to word
            predicted_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    predicted_word = word
                    break
            
            predictions.append(predicted_word)
            sequence.append(predicted_index)
        
        return " ".join(predictions)
        
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Main app
def main():
    st.markdown('<h1 class="main-header">🔮 Next Word Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model, tokenizer = load_components()
    
    if model is None:
        st.error("Failed to load model. Please check if model.h5 is in the correct directory.")
        return
        
    if tokenizer is None:
        st.error("Failed to load tokenizer. Please check if tokenizer.pkl is in the correct directory.")
        return
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area(
            "Input Text", 
            "The weather today is",
            height=150,
            help="Enter some text to start the prediction"
        )
        
        # Prediction parameters
        col_a, col_b = st.columns(2)
        with col_a:
            num_words = st.slider(
                "Words to predict", 
                min_value=1, 
                max_value=10, 
                value=3,
                help="Number of words to generate"
            )
        with col_b:
            temperature = st.slider(
                "Temperature", 
                min_value=0.1, 
                max_value=2.0, 
                value=1.0, 
                step=0.1,
                help="Higher values = more creative, Lower values = more predictable"
            )
        
        # Predict button
        if st.button("Predict Next Words", type="primary"):
            with st.spinner("Generating prediction..."):
                time.sleep(0.5)  # Simulate processing
                prediction = predict_next_words(input_text, num_words, temperature)
                
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                st.success(f"**{input_text} {prediction}**")
                st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Examples to try")
        examples = [
            "I want to eat",
            "Machine learning is",
            "The future of AI",
            "In the beginning",
            "She went to the",
            "The best way to",
            "Artificial intelligence will"
        ]
        
        for example in examples:
            if st.button(example, key=example):
                st.session_state.input_text = example
        
        st.markdown("---")
        st.info("💡 **Tip**: Adjust the temperature slider to control the creativity of predictions.")
        
        # Model info
        with st.expander("Model Information"):
            st.write(f"**Model Architecture**: {model.name}")
            st.write(f"**Input Shape**: {model.input_shape}")
            st.write(f"**Output Shape**: {model.output_shape}")
            st.write(f"**Vocabulary Size**: {len(tokenizer.word_index)}")

if __name__ == "__main__":
    main()