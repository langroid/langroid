from langroid.utils.constants import NO_ANSWER

EXTRACT_RELEVANT = """
    Here is a passage from a long document, followed by a question. 
    In case the passage contains any text relevant to answer the question, return it 
    verbatim.
    {passage}    
    Question:{question}
    Relevant text, if any: """.strip()

EXTRACTION_PROMPT_GPT4 = f"""
Given the content and question below, extract COMPLETE SENTENCES OR PHRASES 
VERBATIM from the content, that are relevant to answering the question (if such text 
exists), even if it contradicts your knowledge, and even if it is factually incorrect.
Do not  make up an answer that is not supported by the content. 
When you answer, be concise, no need to explain anything. If there is no relevant text,
simply say {NO_ANSWER}.

Content: {{content}}
Question: {{question}}
Relevant text, if any:
"""

EXTRACTION_PROMPT = f"""
    Given the content and question below, extract a COMPLETE SENTENCE verbatim from the 
    content that is relevant to answering the question (if such text exists). Do not 
    make up an answer.
    
    Content: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in
    Paris, France. It is named after Gustave Eiffel, whose company designed and built
    the tower. It is a recognizable landmark.
    Question: Where is the Eiffel Tower located?
    Relevant text, if any: on the Champ de Mars in Paris, France.
    
    Content: Apples and oranges are both fruits, but differ in taste and texture.
    Apples are sweet and crisp, while oranges are citrusy and juicy. Both are
    nutritious and commonly consumed worldwide.
    Question: What are the similarities between apples and oranges?
    Relevant text, if any: both fruits
    
    Content: The sun rises in the east and sets in the west. It is a source of light
    and warmth for the Earth.
    Question: What is the color of the sun?
    Relevant text, if any: {NO_ANSWER}
    
    Content: {{content}}
    Question: {{question}}
    Relevant text (COMPLETE SENTENCE), if any:
    """.strip()

SUMMARY_ANSWER_PROMPT_GPT4 = f"""

        Use the provided extracts (with sources)  to answer the question. 
        If there's not enough information, respond with {NO_ANSWER}. Use only the 
        information in these extracts, even if your answer is factually incorrect, 
        and even if the answer contradicts other parts of the document. The only 
        important thing is that your answer is consistent with and supported by the 
        extracts. Compose your complete answer and cite all supporting sources on a 
        separate separate line as follows. For each SOURCE, (a) show the source name,
        and (b) the exact EXTRACT from the source that supports your answer.
        DO NOT SHOW THE FULL EXTRACT; only show the FIRST 3 words and LAST 3 words.  
        For example:
        SOURCE:  https://en.wikipedia.org/wiki/Tree
        EXTRACTS: "The tree species ... oak, maple, and birch."
        
        SOURCE: https://en.wikipedia.org/wiki/Oak
        EXTRACTS: "The oak trees ... longevity and strength."
        
        {{extracts}}
        
        {{question}}
        Answer:   
""".strip()

ANSWER_PROMPT_USE_HISTORY_GPT4 = f"""

        Use ANY of the information earlier, as well as the extracts provided below 
        (with sources)  to answer the question. If there's not 
        enough information, respond with {NO_ANSWER}.
        Use only the information in this conversation or these extracts, 
        even if your answer is factually incorrect, and even 
        if the answer contracts other parts of the document.
        The only important thing is that your answer is 
        consistent with information provided here or earlier.
        Compose your complete answer and cite all supporting sources 
        on a separate separate line as "SOURCE:". 
        When citing a SOURCE: be concise, whether it refers to a source in these 
        extracts, or info provided earlier.
        
        {{extracts}}
        {{question}}
        Answer:   
""".strip()


SUMMARY_ANSWER_PROMPT = f"""
        Use the provided extracts (with sources)  to answer the question. 
        If there's not enough information, respond with {NO_ANSWER}.
        Use only the information in these extracts, even if it contradicts your prior 
        knowledge. Justify your answer by citing your sources, as in these examples:
        
        Extract: The tree species in the garden include oak, maple, and birch.
        Source: https://en.wikipedia.org/wiki/Tree
        Extract: The oak trees are known for their longevity and strength.
        Source: https://en.wikipedia.org/wiki/Oak
        Question: What types of trees are in the garden?
        Answer: The types of trees in the garden include oak, maple, and birch.
        SOURCE: https://en.wikipedia.org/wiki/Tree
        TEXT: The tree species in the garden include oak, maple, and birch.
        
        Extract: The experiment involved three groups: control, low dose, and high 
        dose. 
        Source: https://en.wikipedia.org/wiki/Experiment
        Extract: The high dose group showed significant improvement in symptoms.
        Source: https://en.wikipedia.org/wiki/Experiment
        Extract: The control group did not receive any 
        treatment and served as a baseline.
        Source: https://en.wikipedia.org/wiki/Experiment
        Question: How many groups were involved which group showed significant 
        improvement? 
        Answer: There were three groups and the high dose group showed significant 
        improvement in symptoms.
        SOURCE: https://en.wikipedia.org/wiki/Experiment
        TEXT: The experiment involved three groups: control, low dose, and high dose.
        SOURCE: https://en.wikipedia.org/wiki/Experiment
        TEXT: The high dose group showed significant improvement in symptoms.
        
        
        Extract: The CEO announced several new initiatives during the company meeting.
        Source: https://en.wikipedia.org/wiki/CEO
        Extract: The financial performance of the company has been strong this year.
        Source: https://en.wikipedia.org/wiki/CEO
        Question: What new initiatives did the CEO announce?
        Answer: {NO_ANSWER}
        
        {{extracts}}
        {{question}}
        Answer:
        """.strip()
