import numpy as np
from sklearn.decomposition import TruncatedSVD

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    a = set()
    for articles in corpus:
        for word in articles:
            a.add(word)
            
    corpus_words = sorted(list(a))
    num_corpus_words = len(corpus_words)
    # ------------------
    return corpus_words, num_corpus_words


def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    
    # ------------------
    # Write your implementation here.

    #-- Create a num_words by num_words array
    M = np.zeros((num_words, num_words))

    for idx, word in enumerate(words):
        word2Ind[word] = idx
    
    for article in corpus:
        for idx, word in enumerate(article):
            #-- Get window_size words prior to the current word, and 4 words after the current word
            start_idx = idx-window_size
            end_idx = idx+window_size+1

            #-- Need to accomodate special cases near the beginning or end
            if idx - window_size < 0:
                start_idx = 0
            if idx + window_size + 1 > num_words:
                end_idx = num_words

            #-- Don't include the center word itself
            sliding_window = article[start_idx:idx] + article[idx+1:end_idx]
            print(sliding_window)

            for coword in sliding_window:
                #-- The row we care about is the center word's index
                row = word2Ind[word]
                #-- The current coword index is:
                column = word2Ind[coword]
                print("row = " + str(row) + ", column = " + str(column))

                #-- Increase this element in M by 1:
                M[row, column] = M[row, column] + 1

    # ------------------

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of corpus words, number of number of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.transform(M)

    # ------------------

    print("Done.")
    return M_reduced

def main():
    # b = np.array([[1,2,3],[4,5,6]])
    # print(b.shape)

    corpus = ["START All that glitters isn't gold END".split(" "), 
            "START All's well that ends well END".split(" ")]

    corpus_words, num_corpus_words = distinct_words(corpus)
    # print(corpus_words)
    # print(num_corpus_words)

    M, word2Ind = compute_co_occurrence_matrix(corpus, 4)
    print(corpus_words)
    print(M)
    sorted_by_value = sorted(word2Ind.items(), key=lambda kv: kv[1])
    print(sorted_by_value)
    M_reduced = reduce_to_k_dim(M, 2)
    print(M_reduced)


if __name__ == "__main__":
    main()