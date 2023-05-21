# Extracting text from PDF

```python
# Importing necessary libraries

from PyPDF2 import PdfFileReader

def extract_text_from_pdf(file_path):
pdf = PdfFileReader(open(file_path, "rb"))
text = ''
for page in range(pdf.getNumPages()):
text += pdf.getPage(page).extract_text()
return text

# Extract text from the PDF

pdf_text = extract_text_from_pdf('/mnt/data/Gasperov-2021-survey-DRL-MM.pdf')

# Displaying first 1000 characters to get an idea of the extracted text.
pdf_text[:1000]  
```

# Summarizing the extracted text

```python
# Importing necessary libraries for text summarization
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Summarize the extracted text
summary_text = summarizer(pdf_text, max_length=500, min_length=30, do_sample=False)

summary_text
```

Or alternatively, you can use the `gensim` library for summarization.

```python
# Importing necessary library for text summarization
from gensim.summarization import summarize

# Summarize the extracted text
summary_text = summarize(pdf_text)

summary_text
```
