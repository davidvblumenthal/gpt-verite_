import nltk
import jsonlines
from itertools import groupby


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


"""
    Helper function that turns list of strings to one string
"""
def flatten_str(l):
    s = ''.join(l)
    return s


def reconstruct_document(doc_sentences: list) -> str:
    return(flatten_str(doc_sentences))


def insert_special_tokens_single(sentence: str, end_sentence_token='<|endofsentence|>') -> str:
    # append  |<SENT_END>| token
    sentence += end_sentence_token

    return sentence


# Initialize
splitter = nltk.load("tokenizers/punkt/english.pickle")
splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(train_text = splitter._params,
                                                      lang_vars = CustomLanguageVars())



def preprocess_loss_mask(document: str) -> str:
    # split document into sentences
    doc_sentences = []
    for sent in splitter.tokenize(document):
        # append special token at the end of the sentence
        sent = insert_special_tokens_single(sent, '<|endofsentence|>')
        # append result
        doc_sentences.append(sent)

    # reconstruct document; flatten list of strings to one string
   
    return reconstruct_document(doc_sentences)



def split_ids_at_endofsentence_token(token_ids: list, special_token_id: int) -> list:  

    sublists = [list(group) for key, group in groupby(token_ids, lambda x: x == special_token_id) if not key]
        
    return sublists




def construct_loss_mask(token_ids_sublists: list, multiple=2) -> list:
    doc_loss_mask = []
    doc_token_ids = []
    # iterate over the sublists, -> they represent one sentence
    for sentence in token_ids_sublists:
        # Get index where to split
        split_idx = len(sentence) // 2 # // -> floor division operator rounds down to the nearest integer
        
        # Different case for even and uneven length
        if len(sentence) % 2 == 0:
            # exactly half so length will match
            # loss_mask is first half
            loss_mask = [1] * split_idx
            # sentence splitter inludes the character following the .
            # therefore add the last loss mask token manually
            sh_loss_mask = [multiple] * (split_idx - 1)
            sh_loss_mask.append(1)

        
        else:
            # due to floor division one idx would get lost
            loss_mask = [1] * split_idx
            # same as above + 1 - 1 kept for clarity
            sh_loss_mask = [multiple] * (split_idx + 1 - 1)
            sh_loss_mask.append(1)
        
        # get the final loss mask by extending the two lists together
        loss_mask.extend(sh_loss_mask)

        assert len(sentence) == len(loss_mask), "loss_mask and sentence should have same length"

        doc_loss_mask.extend(loss_mask)
        doc_token_ids.extend(sentence)

    
    assert len(doc_loss_mask) == len(doc_token_ids), "loss_mask and sentence should have same length"


    return doc_token_ids, doc_loss_mask



# Function to help debugging tokenization

def write_tokenized_text_to_file(tokenized_text: dict, file_path: str):
   with jsonlines.open(file_path, mode='a') as writer:
        writer.write(tokenized_text)