import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        except Exception as e:
            st.error(f"Failed to load model or tokenizer: {e}")
            raise e
        
    def summarize_text(self, text: str) -> str:
        """
        Summarizes the input text using the fine-tuned T5 model.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The summarized text.
        """
        if not text.strip():
            return "Error: Input text is empty. Please provide valid text."

        try:
            inputs = self.tokenizer.encode(
                "summarize: " + text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            summary_ids = self.model.generate(
                inputs,
                max_length=128,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"
    
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
            try:
                text = uploaded_file.read().decode("utf-8")
                st.write("Text uploaded for summarization:")
                st.write(text)
                if st.button("Summarize Upload Text"):
                    summary = self.summarize_text(text)
                    st.write("Summary:")
                    st.write(summary)
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

        # Option to manually input text
        text_input = st.text_area("Or manually input text")
        if st.button("Summarize Input Text"):
            summary = self.summarize_text(text_input)
            st.write("Summary:")
            st.write(summary)

if __name__ == "__main__":
    best_model_path = '../fine_tuned/fine_tuned_t5'
     
    if best_model_path.strip():
        app = TextSummarizationApp(best_model_path)
        app.run()
    else:
        st.error("Please specify the path to the model checkpoint.")
