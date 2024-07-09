import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForQuestionAnswering
import torch

# using the BART model for summarization
# according to huggingface documentation, the BART model is one of the best for summarization tasks
# the model is trained on CNN/DailyMail dataset
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

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


    def initUI(self):
        self.setWindowTitle('News Article Summarizer')

        layout = QVBoxLayout()

        self.url_label = QLabel('Enter the news Article URL:')
        layout.addWidget(self.url_label)

        self.url_input = QLineEdit(self)
        layout.addWidget(self.url_input)

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
        #-------------------------------------------
        self.setLayout(layout)

    def summarize_article(self):
        url = self.url_input.text()
        article_text = self.get_article_text(url)
        if article_text:
            summary = self.summarize_text(article_text)
            self.summary_output.setText(summary)
        else:
            self.summary_output.setText("Could not fetch article text. Please check the URL and try again.")
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
        # encode the text  and set the max length to 1024
        inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)
        # generate the summary output using beam search
        summary_ids = model.generate(inputs, max_length=180, min_length=80, length_penalty=2.0, num_beams=4,
                                     early_stopping=True)
        # decode the summary output and remove the special tokens
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    # Function to answer the question based on the article text
    def answer_question(self):
        question = self.question_input.text()
        context = self.summary_output.toPlainText()
        if question and context:
        #if question and self.article_text:
            #answer = self.get_answer(question, self.article_text)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SummarizerApp()
    ex.resize(1000, 800)
    ex.show()
    sys.exit(app.exec_())
