
'''
To allow for a sliding window splitting with an overlap of `k` tokens between \
        consecutive chunks, you can modify the chunking approach as follows:

1. Maintain a buffer (a list) to store tokens that belong to the current chunk.
2. For each sentence in the document, tokenize the sentence into subwords and append
the tokens to the buffer.
3. If the buffer exceeds the maximum tokens per chunk, take the first
`max_tokens_per_chunk` tokens from the buffer and add them as a chunk to the list of chunks.
4. Then, remove the first `max_tokens_per_chunk - k` tokens from the buffer to create
an overlap of `k` tokens for the next chunk.
5. Continue this process until all tokens have been processed and the buffer is empty.

Here's the modified code to implement the sliding window splitting approach with a
specified overlap:
'''

from transformers import GPT2Tokenizer
import nltk

def chunk_document(document, max_tokens_per_chunk, overlap_tokens):
    # Load the GPT-2 tokenizer (similar to GPT-3's tokenizer)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Split the document into paragraphs
    paragraphs = document.split('\n\n')

    # Initialize an empty list to store the chunks
    chunks = []
    token_buffer = []

    # Iterate over the paragraphs
    for paragraph in paragraphs:
        # Tokenize the paragraph into sentences
        sentences = nltk.sent_tokenize(paragraph)

        # Iterate over the sentences
        for sentence in sentences:
            # Tokenize the sentence into subwords
            sentence_tokens = tokenizer.tokenize(sentence)

            # Add tokens to the buffer
            token_buffer.extend(sentence_tokens)

            # If the buffer exceeds the maximum tokens per chunk, create a chunk and
            # slide the window
            while len(token_buffer) >= max_tokens_per_chunk:
                chunk = token_buffer[:max_tokens_per_chunk]
                chunks.append(tokenizer.convert_tokens_to_string(chunk))
                token_buffer = token_buffer[max_tokens_per_chunk - overlap_tokens:]

    # Add the remaining tokens in the buffer as the last chunk
    if token_buffer:
        chunks.append(tokenizer.convert_tokens_to_string(token_buffer))

    return chunks

# Example usage
# document = "Your long document text goes here. The document may contain paragraphs and sentences."
# max_tokens_per_chunk = 50
# overlap_tokens = 10  # Overlap of 10 tokens between consecutive chunks
# chunks = chunk_document(document, max_tokens_per_chunk, overlap_tokens)
#
# # Printing the chunks
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n")


'''
In this code, we use a `token_buffer` to temporarily store tokens for the current 
chunk. When the number of tokens in the buffer exceeds the specified maximum tokens 
per chunk, we create a chunk from the buffer and then slide the window by removing 
the first `max_tokens_per_chunk - overlap_tokens` tokens from the buffer. The 
remaining tokens in the buffer will overlap with the next chunk. The process 
continues until all tokens are processed, and the last chunk is created from the 
remaining tokens in the buffer.      
'''