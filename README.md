# Text Summarization with T5ForConditionalGeneration
This project provides a simple summarizer using fine-tuned T5ForConditionalGeneration model to generate text summarice

## Table of Contents
- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## About
The Text Summarization Project aims to develop an abstractive summarization model using the T5 architecture, fine-tuned on the California state bill subset of the BillSum dataset. The primary goal is to create a tool that can efficiently summarize lengthy texts from newsletters and other sources, providing concise and informative summaries.

Key features:
- Upload a text file for summarization.
- Input custom text and generate summaries in real-time.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/martinp95/text_summarization_with_T5ForConditionalGeneration.git
    cd text_summarization_with_T5ForConditionalGeneration
    ```

2. Create and activate the Conda environment:
    ```sh
    conda env create -f environment.yml
    conda activate text_summarization_with_T5ForConditionalGeneration
    ```

## Usage
### Step 1: Fine-Tune the Model
To fine-tune the model, use the Jupyter notebook `train_model.ipynb`. This notebook walks you through the steps of training the T5 model on the desired dataset.

1. Open the notebook

2. Execute the cells sequentially to fine-tune the model.

3. Save the fine-tune model in the `fine_tuned/` direcotry

### Step 2: Run the Application
Once the model is fine-tuned and saved, you can launch the Streamlit application:

1. Start the application:
```sh
cd ./src/
streamlit run app.py
```

2. Open the local URL displayed in the terminal in your browser.
![Summarization APP](/images/text_summarization_application.png)

### Step 3: Using the Application
* Upload a Text File: Upload a `.txt` file containing the text you want to summarize.
![Uploaded File Summarization](/images/upload_file_summarization.png)

* Manual Text Input: Use the input field to type or parte text directly
![Manual Text Summarization](/images/manually_input_summarize.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.