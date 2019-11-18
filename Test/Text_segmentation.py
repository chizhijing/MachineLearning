import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


if __name__=="__main__":

    doc="An improved weighted K-means clustering algorithm is proposed.\
    In the presented algorithm,firstly the original clustering center is obtained\
    according to the data sample distribution. Then the improved K-means algorithm\
    using the feature weights is designed. Experiment results have shown that the proposed\
    clustering algorithm can produce high quality clustering steadily and deal with the\
    symbolic data as well as the numeric data."


    print("文本分句")
    tokens = nltk.sent_tokenize(doc)
    for i, token in enumerate(tokens):
        print(i, token)

    print("文本分词")
    tokens=nltk.word_tokenize(doc)
    for i,token in enumerate(tokens):
        if token in stopwords.words('english'):
            print("Stop Word:",i,token)
        else:
            print("Norm Word:",i,token)


