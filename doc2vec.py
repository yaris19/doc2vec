import random

from Bio import Entrez, Medline
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

Entrez.email = "YJ.vanThiel@student.han.nl"


def retrieve_pubmed_articles(abstracts_file_name, pmids_file_name, seed=4):
    pmids = random.Random(seed).sample(range(1, 33500000), 20000)
    pmids_retr = ', '.join(map(str, pmids))
    used_pmids = []

    abstract_count = 0
    with open(abstracts_file_name, "w", encoding="utf-8") as outfile:
        for index, start in enumerate(range(0, len(pmids), 10000)):
            print("batch", index + 1)
            handle = Entrez.efetch(db='pubmed', id=pmids_retr,
                                   rettype='medline', retmode='text',
                                   retstart=start)
            records = Medline.parse(handle)

            for record in tqdm(records, total=10000, desc="Progress"):
                if "AB" in record and abstract_count < 10000:
                    outfile.write(record["AB"] + "\n")
                    abstract_count += 1
                    used_pmids.append(record["PMID"])
                elif abstract_count >= 10000:
                    break

            handle.close()
            print("collected abstracts:", abstract_count)

    with open(pmids_file_name, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(used_pmids))
    return used_pmids


def get_data(abstracts_file, pmids_file):
    pmids = []
    with open(pmids_file, "r") as f:
        for line in f:
            pmids.append(int(line.strip()))

    abstracts = []
    with open(abstracts_file, "r") as f:
        for line in f:
            abstracts.append(line.strip())

    data = {}
    for pmid, abstract in zip(pmids, abstracts):
        data[pmid] = abstract

    return data


def preprocess_abstracts(data):
    documents = []

    for i, (pmid, abstract) in enumerate(data.items()):
        words = abstract.strip().split()
        tags = [str(pmid)]
        documents.append(TaggedDocument(words=words, tags=tags))

    return documents


def doc2vec(documents, model_file):
    print("training the model")
    model = Doc2Vec(documents=documents, vector_size=10, workers=8,
                    epochs=20)
    model.save(model_file)
    print("done training the model")


def get_random_abstract(documents):
    return random.choice(documents)


def predict(documents, model_file):
    print("predicting on random abstract")
    model = Doc2Vec.load(model_file)

    random_abstract = get_random_abstract(documents)
    print(f"PMID of random abstract {random_abstract.tags[0]}")

    vector = model.infer_vector(random_abstract.words)

    similar_articles = model.docvecs.most_similar([vector])
    for similar_article in similar_articles:
        pubmed_index, score = similar_article
        print(pubmed_index, score)

    print("done predicting on random abstract")


if __name__ == "__main__":
    abstracts_file_name = "data/PubMed_abstracts.txt"
    pmids_file_name = "data/PubMed_ids.txt"
    model_file = "data/doc2vec.model"

    retrieve_pubmed_articles(abstracts_file_name, pmids_file_name)

    data = get_data(abstracts_file_name, pmids_file_name)
    documents = preprocess_abstracts(data)

    # train doc2vec
    doc2vec(documents, model_file)

    # predict on random abstract from train set
    predict(documents, model_file)
