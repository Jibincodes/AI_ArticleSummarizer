import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, \
    QMessageBox, QRadioButton, QHBoxLayout, QButtonGroup
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForQuestionAnswering, LongformerTokenizer, LongformerModel
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import wrap
# for extracting from PDF
import fitz

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

class SummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.pdf_text = ""

    # Setting up the UI components
    def initUI(self):
        self.setWindowTitle('News Article Summarizer')

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
        #-----------------------------------------------
        #spliting the text into chunks of 4096 tokens
        chunk_size = 4096
        overlap = 512 #to maintain the context
        tokens = longformer_tokenizer.encode(text)
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap)]

        summaries = []
        for chunk in chunks:
            #convert the chunk to a tensor
            #inputs = torch.tensor(chunk).unsqueeze(0)
            #with torch.no_grad():
            #    outputs = longformer_model(inputs)

            #-----------------------------------------------
            #get the summary using BART model
            chunk_text = longformer_tokenizer.decode(chunk, skip_special_tokens=True)
            bart_inputs = tokenizer.encode("summarize: " + chunk_text, return_tensors='pt', max_length=1024, truncation=True)
             # generate the summary output using beam search
            summary_ids = model.generate(bart_inputs, max_length=400, min_length=200, length_penalty=2.0, num_beams=4,
                                     early_stopping=True)
            # decode the summary output and remove the special tokens
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        combined_summary = ' '.join(summaries)
        # Now summarize the combined summary text
        #bart_inputs = tokenizer.encode("summarize: " + combined_summary, return_tensors='pt', max_length=1024,
        #                               truncation=True)
        #summary_ids = model.generate(bart_inputs, max_length=600, min_length=300, length_penalty=2.0, num_beams=8,
        #                             early_stopping=True)
        #final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return combined_summary

    # Function to answer the question based on the article text
    def answer_question(self):
        #url = self.url_input.text()
        #article_text = self.get_article_text(url)
        question = self.question_input.text()
        context = self.summary_output.toPlainText()
        if question and context:
        #if question and article_text:
            #answer = self.get_answer(question, article_text)
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
    def save_pdf(self, file_path, summary):
     try:
        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter
        c.drawString(100, height - 100, "Summary:")
        text = c.beginText(100, height - 120)
        text.setFont("Times-Roman", 12)
        wrapped_text = wrap(summary, 80)  # Wrap text at 80 characters
        for line in wrapped_text:
            text.textLine(line)
        c.drawText(text)
        c.save()
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

    #function to extract the text from the PDF file
    def get_pdf_text(self, file_path):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")  # Get text blocks

                # Sort blocks by their vertical and then horizontal positions
                blocks.sort(key=lambda b: (b[1], b[0]))

                for block in blocks:
                    text += block[4]  # Extract the actual text from the block
                    text += "\n"  # Add a newline to separate blocks

                text += "\n"  # Add a newline to separate pages

            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SummarizerApp()
    ex.resize(1000, 800)
    ex.show()
    sys.exit(app.exec_())
