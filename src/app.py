import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration


class TextSummarizationApp:
    """
    A Streamlit-based application for text summarization using a fine-tuned T5 model.

    Attributes:
        model_path (str): Path to the fine-tuned T5 model checkpoint.
        model (T5ForConditionalGeneration): The fine-tuned T5 model.
        tokenizer (T5Tokenizer): Tokenizer associated with the T5 model.
    """

    def __init__(self, model_path: str):
        """
        Initializes the app with the fine-tuned T5 model and tokenizer.

        Args:
            model_path (str): Path to the fine-tuned T5 model checkpoint.
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
        Summarizes the given text using the fine-tuned T5 model.

        Args:
            text (str): Input text to summarize.

        Returns:
            str: Summarized text or error message if the input is invalid.
        """
        if not text.strip():
            return "Error: Input text is empty. Please provide valid text."

        try:
            # Preprocess input text for summarization
            inputs = self.tokenizer.encode(
                "summarize: " + text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            # Generate the summary
            summary_ids = self.model.generate(
                inputs,
                max_length=128,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            return f"Error generating summary: {e}"

    def run(self):
        """
        Launches the Streamlit application.
        """
        st.title("Text Summarization Application")
        st.write("""
        Use this application to summarize text using a fine-tuned T5 model.
        You can upload a text file or manually input text to generate a summary.
        """)

        # Option 1: Upload a text file
        uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
        if uploaded_file is not None:
            try:
                text = uploaded_file.read().decode("utf-8")
                st.subheader("Uploaded Text")
                st.write(text)

                if st.button("Summarize Uploaded Text"):
                    summary = self.summarize_text(text)
                    st.subheader("Generated Summary")
                    st.write(summary)
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")

        # Option 2: Manually input text
        st.subheader("Manually Input Text")
        text_input = st.text_area("Enter text here:")
        if st.button("Summarize Input Text"):
            summary = self.summarize_text(text_input)
            st.subheader("Generated Summary")
            st.write(summary)


if __name__ == "__main__":
    # Specify the path to the fine-tuned model
    best_model_path = '../fine_tuned/fine_tuned_t5'

    if best_model_path.strip():
        app = TextSummarizationApp(best_model_path)
        app.run()
    else:
        st.error("Please specify a valid path to the model checkpoint.")
