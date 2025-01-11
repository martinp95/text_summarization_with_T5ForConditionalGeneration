import streamlit as st
from transformers import T5Tokenizer
from training import Summarizer

class TextSummarizationApp:
    """
    A Streamlit application for text summarization using a fine-tuned T5 model.
    
    Attributes:
        model_path (str): The path to the fine-tuned T5 model checkpoint.
        model (Summarizer): The fine-tuned T5 model for text summarization.
        tokenizer (T5Tokenizer): The tokenizer for the T5 model.
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the TextSummarizationApp with the fine-tuned T5 model and tokenizer.

        Args:
            model_path (str): The path to the fine-tuned T5 model checkpoint.
        """
        self.model_path = model_path
        self.model = Summarizer.load_from_checkpoint(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
    def summarize_text(self, text: str) -> str:
        """
        Summarizes the input text using the fine-tuned T5 model.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The summarized text.
        """
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.model.generate(inputs, max_length=128, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def run(self):
        """
        Runs the Streamlit application for text summarization.
        """
        st.title("Text Summarization Application")
        st.write("""
        This application allows you to summarize text using a pre-trained T5 model.
        You can either upload a text file or manually input text to get a summary.
        """)
        
        # Option to upload a text file
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")
            st.write("Text uploaded for summarization:")
            st.write(text)
            if st.button("Summarize Upload Text"):
                summary = self.summarize_text(text)
                st.write("Summary:")
                st.write(summary)
                
        # Option to manually input text
        text_input = st.text_area("Or manually input text")
        if st.button("Summarize Input Text"):
            summary = self.summarize_text(text_input)
            st.write("Summary:")
            st.write(summary)
            
if __name__ == "__main__":
    app = TextSummarizationApp("")
    app.run()