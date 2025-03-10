
---

# **Using `marker` as a PDF Parser in `langroid`**  

## **Installation**  

### **Standard Installation**  
To use [`marker`](https://github.com/VikParuchuri/marker) as a PDF parser in `langroid`, 
install it with the `marker-pdf` extra:

```bash
pip install langroid[marker-pdf]
```
or in combination with other extras as needed, e.g.:
```bash
pip install "langroid[marker-pdf,hf-embeddings]"
```

Note, however, that due to an **incompatibility with `docling`**,
if you install `langroid` using the `all` extra 
(or another extra such as  `doc-chat` or `pdf-parsers` that 
also includes `docling`),
e.g. `pip install "langroid[all]"`, or `pip install "langroid[doc-chat]"`,
then due to this version-incompatibility with `docling`, you will get an 
**older** version of `marker-pdf`, which does not work with Langroid.
This may not matter if you did not intend to specifically use `marker`, 
but if you do want to use `marker`, you will need to install langroid
with the `marker-pdf` extra, as shown above, in combination with other
extras as needed, as shown above.


#### **For Intel-Mac Users**  
If you are on an **Intel Mac**, `docling` and `marker` cannot be 
installed together with langroid as extras, 
due to a **transformers version conflict**.  
To resolve this, manually install `marker-pdf` with:  

```bash
pip install marker-pdf[full]
```

Make sure to install this within your `langroid` virtual environment.

---

## **Example: Parsing a PDF with `marker` in `langroid`**  

```python
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import MarkerConfig, ParsingConfig, PdfParsingConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")

# Path to your PDF file
path = "<path_to_your_pdf_file>"

# Define parsing configuration
parsing_config = ParsingConfig(
    n_neighbor_ids=2,  # Number of neighboring sections to keep
    pdf=PdfParsingConfig(
        library="marker",  # Use `marker` as the PDF parsing library
        marker_config=MarkerConfig(
            config_dict={
                "use_llm": True,  # Enable high-quality LLM processing
                "gemini_api_key": gemini_api_key,  # API key for Gemini LLM
            }
        )
    ),
)

# Create the parser and extract the document
marker_parser = DocumentParser.create(path, parsing_config)
doc = marker_parser.get_doc()
```

---

## **Explanation of Configuration Options**  

If you want to use the default configuration, you can omit `marker_config` entirely.

### **Key Parameters in `MarkerConfig`**
| Parameter        | Description |
|-----------------|-------------|
| `use_llm`       | Set to `True` to enable higher-quality processing using LLMs. |
| `gemini_api_key` | Google Gemini API key for LLM-enhanced parsing. |



You can further customize `config_dict` by referring to [`marker_pdf`'s documentation](https://github.com/VikParuchuri/marker/blob/master/README.md).  

Alternatively, run the following command to view available options:  

```sh
marker_single --help
```

This will display all supported parameters, which you can pass as needed in `config_dict`.

---