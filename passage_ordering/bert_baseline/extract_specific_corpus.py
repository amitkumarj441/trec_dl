from passage_ordering.bert_baseline import convert_treccar_to_tfrecord as tool, trec_car_classes
import time
import json


def get_passage_ids(topic_num=100000):
    qrels_dict = tool.load_qrels(r'E:\PythonProject\trec_dl\passage_ordering\data\tf_data\test.qrels')
    run_dict = tool.load_run(r'E:\PythonProject\trec_dl\passage_ordering\data\tf_data\test.run')

    topics_set = set(qrels_dict.keys()).intersection(set(run_dict.keys()))
    while len(topics_set) > topic_num:
        topics_set.pop()

    passage_ids = []

    for topic in topics_set:
        passage_ids.extend(qrels_dict[topic])
        passage_ids.extend(run_dict[topic])

    passage_ids = set(passage_ids)
    print("try to load {} passages. ".format(len(passage_ids)))
    return passage_ids, topics_set


def load_corpus(path, passage_ids):
    """Loads TREC-CAR's paraghaphs into a dict of key: title, value: paragraph."""
    corpus = {}
    start_time = time.time()
    APPROX_TOTAL_PARAGRAPHS = 30000000
    num_ignored_passage = 0
    num_loaded_passage = 0
    num_total_passage = 0

    with open(path, 'rb') as f:
        for i, p in enumerate(trec_car_classes.iter_paragraphs(f)):
            num_total_passage = i
            if p.para_id not in passage_ids:
                num_ignored_passage += 1
                continue
            num_loaded_passage += 1
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

    print("Have ignored {} passages.".format(num_ignored_passage))
    print("Have loaded {} passages.".format(num_loaded_passage))
    print("total {} passages.".format(num_total_passage))
    return corpus


if __name__ == '__main__':
    ids_set, topics_set = get_passage_ids()
    with open('topics.txt', 'w') as f:
        for topic in topics_set:
            f.write(topic+"\n")
    print("========== Finish saving topics ===========")

    d = load_corpus(r'E:\SZTY\TREC-CAR\paragraphCorpus.v2.0\dedup.articles-paragraphs.cbor', ids_set)
    with open('corpus1.json', 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)
    # with open('test.json', 'r', encoding='utf-8') as f:
    #     print(json.load(f))
