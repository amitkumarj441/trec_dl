"""Converts TREC-CAR queries and corpus into TFRecord that will be consumed by BERT.

The main necessary inputs are:
- Paragraph Corpus (CBOR file) 
- Pairs of Query-Relevant Paragraph (called qrels in TREC's nomenclature)
- Pairs of Query-Candidate Paragraph (called run in TREC's nomenclature)

The outputs are 3 TFRecord files, for training, dev and test.
"""
import collections
import json
import os
import re
import tensorflow as tf
import time
# local modules
from passage_ordering.bert_baseline import tokenization
from passage_ordering.bert_baseline import trec_car_classes

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_folder", "../data/tf_data/",
    "Folder where the TFRecord files will be writen.")

flags.DEFINE_string(
    "vocab_file",
    "../data/tf_data/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "corpus", r"E:\SZTY\TREC-CAR\test200\test200-train\fold-0-train.pages.cbor-paragraphs.cbor",
    "Path to the cbor file containing the Wikipedia paragraphs.")

flags.DEFINE_string(
    "corpus_json", r"E:\PythonProject\trec_dl\passage_ordering\bert_baseline\corpus1.json",
    "Path to the json file containing the 469K Wikipedia paragraphs.")

flags.DEFINE_string(
    "qrels_train", "../data/tf_data/train.qrels",
    "Path to the topic / relevant doc ids pairs for training.")

flags.DEFINE_string(
    "qrels_dev", "../data/tf_data/dev.qrels",
    "Path to the topic / relevant doc ids pairs for dev.")

flags.DEFINE_string(
    "qrels_test", "../data/tf_data/test.qrels",
    "Path to the topic / relevant doc ids pairs for test.")

flags.DEFINE_string(
    "run_train", "../data/tf_data/train.run",
    "Path to the topic / candidate doc ids pairs for training.")

flags.DEFINE_string(
    "run_dev", "../data/tf_data/dev.run",
    "Path to the topic / candidate doc ids pairs for dev.")

flags.DEFINE_string(
    "run_test", "../data/tf_data/test.run",
    "Path to the topic / candidate doc ids pairs for test.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_train_docs", 10,
    "The number of docs per query for the training set.")

flags.DEFINE_integer(
    "num_dev_docs", 10,
    "The number of docs per query for the development set.")

flags.DEFINE_integer(
    "num_test_docs", 1000,
    "The number of docs per query for the test set.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_candidate_docs", 8,
    "The number of candidate (negative) documents.")


def get_query_ids(query, tokenizer):
    query = query.replace('enwiki:', '')
    query = query.replace('%20', ' ')
    query = query.replace('/', ' ')
    query = tokenization.convert_to_unicode(query)
    q_ids = tokenization.convert_to_bert_input(
        text=query,
        max_seq_length=FLAGS.max_query_length,
        tokenizer=tokenizer,
        add_cls=True)
    return q_ids


def convert_dataset(data, corpus, set_name, tokenizer):
    output_path = FLAGS.output_folder + '/dataset8_' + set_name + '.tf'

    print('Converting {} to tfrecord'.format(set_name))
    start_time = time.time()

    random_title = list(corpus.keys())[0]

    all_doc_ids = set(corpus.keys())

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for i, query in enumerate(data):
            qrels, doc_titles = data[query]
            # if len(set(qrels) - set(all_doc_ids)) != 0:
            #     continue
            #
            # if len(set(doc_titles) - set(all_doc_ids)) != 0:
            #     continue

            if i % 1000 == 0:
                print('query', query)

            # get query ids
            query_ids = get_query_ids(query, tokenizer)

            query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))

            if set_name == 'train':
                max_docs = FLAGS.num_train_docs
            elif set_name == 'dev':
                max_docs = FLAGS.num_dev_docs
            elif set_name == 'test':
                max_docs = FLAGS.num_test_docs

            doc_titles = doc_titles[:max_docs]

            # Add fake docs so we always have max_docs per query.
            doc_titles += max(0, max_docs - len(doc_titles)) * [random_title]

            docs_length = FLAGS.max_seq_length - len(query_ids)
            query_ids_set = set(query_ids)

            # generate all candidate docs ids into "one" list
            candidate_token_ids = []
            num_candidate_docs = 0
            for candidate_title in doc_titles:
                # skip the positive docs
                if candidate_title in query_ids_set:
                    continue
                else:
                    padded_title = candidate_title
                candidate_token_ids.extend(
                    tokenization.convert_to_bert_input(
                        text=tokenization.convert_to_unicode(corpus[candidate_title]),
                        max_seq_length=docs_length,
                        tokenizer=tokenizer,
                        add_cls=False)
                )
                num_candidate_docs += 1

                if num_candidate_docs == FLAGS.num_candidate_docs:
                    break

            while num_candidate_docs < FLAGS.num_candidate_docs:
                candidate_token_ids.extend(
                    tokenization.convert_to_bert_input(
                        text=tokenization.convert_to_unicode(corpus[padded_title]),
                        max_seq_length=docs_length,
                        tokenizer=tokenizer,
                        add_cls=False)
                )


            # generate all qrels docs ids
            qrels_token_ids = [
                tokenization.convert_to_bert_input(
                    text=tokenization.convert_to_unicode(corpus[qrel_title]),
                    max_seq_length=docs_length,
                    tokenizer=tokenizer,
                    add_cls=False)
                for qrel_title in qrels
            ]

            # loop qrels ids
            for qrel_token_id in qrels_token_ids:
                doc_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=candidate_token_ids))

                qrel_token_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=qrel_token_id))

                paragraph_length = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[docs_length]))

                len_gt_titles_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(qrels)]))

                features = tf.train.Features(feature={
                    'query_ids': query_ids_tf,
                    'doc_ids': doc_ids_tf,
                    'qrel_ids': qrel_token_tf,
                    'len_docs': paragraph_length,
                    'len_gt_titles': len_gt_titles_tf,
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())

            if i % 1000 == 0:
                print('wrote {}, {} of {} queries'.format(set_name, i, len(data)))
                time_passed = time.time() - start_time
                est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
                print('estimated total hours to save: {}'.format(est_hours))


def load_qrels(path, limited_topics=None):
    """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, relevance = line.rstrip().split(' ')
            if limited_topics and topic not in limited_topics:
                continue
            if int(relevance) >= 1:
                qrels[topic].add(doc_title)
            if i % 1000 == 0:
                print('Loading qrels {}'.format(i))
    return qrels


def load_run(path, limited_topics=None):
    """Loads run into a dict of key: topic, value: list of candidate doc ids."""

    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, rank, _, _ = line.split(' ')
            if limited_topics and topic not in limited_topics:
                continue
            if topic not in run:
                run[topic] = []
            run[topic].append((doc_title, int(rank)))
            if i % 100000 == 0:
                # break
                print('Loading run {}'.format(i))
    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for topic, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[topic] = doc_titles

    return sorted_run


def load_corpus_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        cj = json.load(f)
    return cj


def load_corpus(path):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    corpus = {}
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30
    with open(path, 'rb') as f:
        for i, p in enumerate(trec_car_classes.iter_paragraphs(f)):
            para_txt = [elem.text if isinstance(elem, trec_car_classes.ParaText)
                        else elem.anchor_text
                        for elem in p.bodies]

            corpus[p.para_id] = ' '.join(para_txt)
            if i % 10000 == 0:
                print('Loading paragraph {} of {}'.format(i, APPROX_TOTAL_PARAGRAPHS))
                time_passed = time.time() - start_time
                hours_remaining = (
                                          APPROX_TOTAL_PARAGRAPHS - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(
                    hours_remaining))

    return corpus


# output: ground true answer, [candidate_doc_ids]
def merge(qrels, run):
    """Merge qrels and runs into a single dict of key: topic,
    value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
    return data


def main():
    with open("topics.txt", "r") as f:
        topics = f.readlines()
    limited_topics = set([topic.strip() for topic in topics])

    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    if not os.path.exists(FLAGS.output_folder):
        os.mkdir(FLAGS.output_folder)

    print('Loading Corpus...')
    # corpus = load_corpus(FLAGS.corpus)
    corpus = load_corpus_json(FLAGS.corpus_json)

    # for set_name, qrels_path, run_path in [
    #     ('train', FLAGS.qrels_train, FLAGS.run_train),
    #     ('dev', FLAGS.qrels_dev, FLAGS.run_dev),
    #     ('test', FLAGS.qrels_test, FLAGS.run_test)]:
    for set_name, qrels_path, run_path in [
        ('test', FLAGS.qrels_test, FLAGS.run_test)]:
        print('Converting {}'.format(set_name))

        qrels = load_qrels(path=qrels_path, limited_topics=limited_topics)
        run = load_run(path=run_path, limited_topics=limited_topics)
        data = merge(qrels=qrels, run=run)

        convert_dataset(data=data,
                        corpus=corpus,
                        set_name=set_name,
                        tokenizer=tokenizer)

    print('Done!')


if __name__ == '__main__':
    main()
