import string
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, \
    QMessageBox, QRadioButton, QHBoxLayout, QButtonGroup
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForQuestionAnswering, LongformerTokenizer, LongformerModel
import torch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# for extracting from PDF
import fitz
import re

# for extractive summarization
import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
#---------------------------------------------------------------
# using the BART model for summarization
# according to huggingface documentation, the BART model is one of the best for summarization tasks
# the model is trained on CNN/DailyMail dataset
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
#----------------------------------------
# using the Longformer model for handling longer texts
longformer_model_name = 'allenai/longformer-base-4096'
longformer_tokenizer = LongformerTokenizer.from_pretrained(longformer_model_name)
longformer_model = LongformerModel.from_pretrained(longformer_model_name)

# using the BERT model for question answering
#bert_model_name = 'distilbert-base-uncased-distilled-squad'
bert_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)
#----------------------------------------

#the model for extractive summarization for the textrank algorithm
model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
class SummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.pdf_text = ""

    # Setting up the UI components
    def initUI(self):
        self.setWindowTitle('AI Summarizer')

        layout = QVBoxLayout()
        #---------------------------------------
        # adding radio buttons to choose between summarization feature for url or pdf
        self.url_radio = QRadioButton('URL')
        self.pdf_radio = QRadioButton('PDF')
        self.url_radio.setChecked(True)

        self.radio_layout = QHBoxLayout()
        self.radio_layout.addWidget(self.url_radio)
        self.radio_layout.addWidget(self.pdf_radio)

        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.url_radio)
        self.radio_group.addButton(self.pdf_radio)

        layout.addLayout(self.radio_layout)

        # adding new buttons to choose between abstractive and extractive summarization
        self.abstractive_radio = QRadioButton('Abstractive summarization')
        self.extractive_radio = QRadioButton('Extractive summarization')
        self.abstractive_radio.setChecked(True)

        self.summarization_radio_layout = QHBoxLayout()
        self.summarization_radio_layout.addWidget(self.abstractive_radio)
        self.summarization_radio_layout.addWidget(self.extractive_radio)

        self.summarization_radio_group = QButtonGroup()
        self.summarization_radio_group.addButton(self.abstractive_radio)
        self.summarization_radio_group.addButton(self.extractive_radio)

        layout.addLayout(self.summarization_radio_layout)
        #---------------------------------------
        self.url_label = QLabel('Enter the news Article URL:')
        layout.addWidget(self.url_label)

        self.url_input = QLineEdit(self)
        layout.addWidget(self.url_input)

        self.upload_button = QPushButton('Upload PDF', self)
        self.upload_button.clicked.connect(self.upload_pdf)
        layout.addWidget(self.upload_button)

        self.summarize_button = QPushButton('Summarize', self)
        self.summarize_button.clicked.connect(self.summarize_article)
        layout.addWidget(self.summarize_button)

        self.summary_label = QLabel('Summary:')
        layout.addWidget(self.summary_label)

        self.summary_output = QTextEdit(self)
        self.summary_output.setReadOnly(True)
        layout.addWidget(self.summary_output)

        #decided to add a question answering feature
        self.question_label = QLabel('Enter the question:')
        layout.addWidget(self.question_label)

        self.question_input = QLineEdit(self)
        layout.addWidget(self.question_input)

        self.answer_button = QPushButton('To Answer', self)
        self.answer_button.clicked.connect(self.answer_question)
        layout.addWidget(self.answer_button)

        self.answer_label = QLabel('Answer:')
        layout.addWidget(self.answer_label)

        self.answer_output = QTextEdit(self)
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        #new export to pdf button
        self.export_button = QPushButton('Export as PDF', self)
        self.export_button.clicked.connect(self.export_to_pdf)
        layout.addWidget(self.export_button)
        #-------------------------------------------
        self.setLayout(layout)

    def summarize_article(self):
     if self.url_radio.isChecked():
        url = self.url_input.text()
        article_text = self.get_article_text(url)
        if article_text:
            summary = self.summarize_text(article_text)
            self.summary_output.setText(summary)
        else:
            self.summary_output.setText("Could not fetch article text. Please check the URL and try again.")
     elif self.pdf_radio.isChecked():
         if self.pdf_text:
             summary = self.summarize_text(self.pdf_text)
             self.summary_output.setText(summary)
         else:
             self.summary_output.setText("Please upload a PDF file to summarize.")


    # Function to fetch the article text from the URL using paragraph tags
    def get_article_text(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            article_text = ' '.join([para.get_text() for para in paragraphs])
            return article_text
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None

    def summarize_text(self, text):
        if self.abstractive_radio.isChecked():
            return self.abstractive_summarize(text)
        else:
            return self.extractive_summary_textrank(text, num_sentences=10)


    def abstractive_summarize(self, text):
        #-----------------------------------------------
        #spliting the text into chunks of 4096 tokens
        chunk_size = 4096
        overlap = 512 #to maintain the context
        tokens = longformer_tokenizer.encode(text)
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap)]

        summaries = []
        for chunk in chunks:

            #-----------------------------------------------
            #get the final summary using BART model
            chunk_text = longformer_tokenizer.decode(chunk, skip_special_tokens=True)
            bart_inputs = tokenizer.encode("summarize: " + chunk_text, return_tensors='pt', max_length=1024, truncation=True)
            summary_ids = model.generate(bart_inputs, max_length=600, min_length=300, length_penalty=2.0, num_beams=4,
                                     early_stopping=True)

            # decode the summary output and remove the special tokens
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        combined_summary = ' '.join(summaries)
        return combined_summary

    #-------------------------------------------------------
    #This is the extractive summarization using the textrank algorithm
    #The model required has been already loaded above
    def preprocess_text(self, text):
        try:
            sentences = sent_tokenize(text)
            return sentences
        except Exception as e:
            print(f"Error in preprocess_text: {e}")
            return []

    def build_similarity_matrix(self, sentences):
        if model1 is None:
            return np.array([])
        try:
            embeddings = model1.encode(sentences)
            similarity_matrix = cosine_similarity(embeddings)

            # Clip values to avoid overflow due to floating point errors
            np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)

            # diagonal elements are set to zero
            np.fill_diagonal(similarity_matrix, 0)

            # Normalizing the similarity matrix
            norm = np.linalg.norm(similarity_matrix, axis=1, keepdims=True)
            norm[norm == 0] = 1  # making sure we don't divide by zero
            normalized_similarity_matrix = similarity_matrix / norm

            # Debug testing
            print("Embeddings shape:", embeddings.shape)
            print("Similarity matrix shape:", similarity_matrix.shape)
            print("Similarity matrix max value:", np.max(similarity_matrix))
            print("Similarity matrix min value:", np.min(similarity_matrix))

            return normalized_similarity_matrix
        except Exception as e:
            print(f"Error in build_similarity_matrix: {e}")
            return np.array([])

    # Extractive summarization function
    def extractive_summary_textrank(self, text, num_sentences=10, max_iter=1000, alpha=0.85):
        try:
            sentences = self.preprocess_text(text)
            if not sentences:
                return "Error in preprocessing text."

            # Filter out the sentences containing URLs, because in pdf there are always unwanted URLS
            url_pattern = re.compile(r'http\S+|www\S+')
            filtered_sentences = [s for s in sentences if not url_pattern.search(s)]

            if not filtered_sentences:
                return "Error: No valid sentences after filtering out URLs."

            similarity_matrix = self.build_similarity_matrix(filtered_sentences)
            if similarity_matrix.size == 0:
                return "Error in building similarity matrix."

            sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
            try:
                scores = nx.pagerank(sentence_similarity_graph, max_iter=max_iter, alpha=alpha)
            except nx.PowerIterationFailedConvergence as e:
                print(f"Power iteration failed to converge: {e}")
                return "Error: Power iteration failed to converge."

            ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(filtered_sentences)), reverse=True)
            summarize_text = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]

            clean_summary = " ".join(summarize_text).replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(
                " ?", "?")
            # Removing any unnecessary spaces
            clean_summary = " ".join(clean_summary.split())

            return clean_summary
        except Exception as e:
            print(f"Error in extractive_summary_textrank: {e}")
            return "Error in extractive summarization."
    #this is the end of extractive summarization

    # Function to answer the question based on the article text
    def answer_question(self):
        #url = self.url_input.text()
        #article_text = self.get_article_text(url)
        question = self.question_input.text()
        context = self.summary_output.toPlainText()
        if question and context:
            answer = self.get_answer(question, context)
            self.answer_output.setText(answer)
        else:
            self.answer_output.setText("Please provide a valid question.")

    # Function to get the answer using the BERT model
    def get_answer(self, question, context):
     try:
        # encode the question and context using the BERT tokenizer
        inputs = bert_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
        #convert the input ids to a list and to get the tokens
        input_ids = inputs['input_ids'].tolist()[0]
        text_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)

        #getting the answer with the help of the BERT model (using attention mask and feed forward)
        outputs = bert_model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        # get the answer by finding the tokens with the highest start and end scores
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = bert_tokenizer.convert_tokens_to_string(bert_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return answer
     except Exception as e:
        print(f"Error getting answer: {e}")
        return "Sorry, I could not find an answer to your question. Please try again."

    #function to export the summary to a pdf file
    def export_to_pdf(self):
        summary = self.summary_output.toPlainText()
        if summary:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Summary as PDF", "",
                                                       "PDF Files (*.pdf);;All Files (*)", options=options)
            if file_path:
                self.save_pdf(file_path, summary)
        else:
            self.answer_output.setText("No summary to export.")

    #function to save the summary as a pdf file
    def save_pdf(self, file_path, summary):
        try:
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            styles = getSampleStyleSheet()
            style = styles['Normal']
            style.leading = 16

            elements = []
            elements.append(Paragraph("Summary:", styles['Title']))
            elements.append(Spacer(1, 0.2 * inch))
            # Adding the summary text
            elements.append(Paragraph(summary, style))
            doc.build(elements)

            QMessageBox.information(self, "PDF Saved", "The summary has been successfully saved as a PDF.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving PDF: {e}")

    #function to upload the pdf file and extract the text
    def upload_pdf(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload PDF", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if file_path:
            self.pdf_text = self.get_pdf_text(file_path)
            if self.pdf_text:
                self.summary_output.setText("PDF uploaded successfully. Click 'Summarize' to get the summary.")
            else:
                self.summary_output.setText("Could not extract text from PDF. Please try again.")

    #function to clean the text
    def clean_text(self, text):
        # Removing multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Removing leading/trailing whitespace
        text = text.strip()
        return text

    #function to extract the text from the PDF file
    def get_pdf_text(self, file_path):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")

                # Sorting the blocks by their vertical and then horizontal positions to get the text in order
                blocks.sort(key=lambda b: (b[1], b[0]))

                for block in blocks:
                    text += block[4]
                    text += "\n"

                text += "\n"  # newline to separate pages

            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

    #function is not used, this is an alternative to the get_pdf_text function
    """def file_preprocessing(self, file):
        try:
            print(f"Loading PDF file: {file}")
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            print(f"Number of pages extracted: {len(pages)}")

            # Initializing the text splitter with the decided chunk size and overlap
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

            # Split documents into chunks
            texts = text_splitter.split_documents(pages)
            print(f"Number of text chunks created: {len(texts)}")

            # Combining chunks into a single string
            final_texts = ""
            for text in texts:
                final_texts += text.page_content
                print(f"Chunk length: {len(text.page_content)}")

            if not final_texts.strip():
                print("Warning: Extracted text is empty.")

            return final_texts

        except FileNotFoundError:
            print(f"Error: File not found {file}.")
            return ""
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return "" """

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SummarizerApp()
    ex.resize(1000, 800)
    ex.show()
    sys.exit(app.exec_())
