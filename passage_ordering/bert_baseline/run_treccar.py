"""Code to train and eval a BERT passage re-ranker on the TREC CAR dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,5,6'
import sys

sys.path.append('../..')

# local modules
from passage_ordering.bert_baseline import metrics
from passage_ordering.bert_baseline import modeling
from passage_ordering.bert_baseline import optimization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir",
    r"../data/tf_data/",
    "The input data dir. Should contain the .tfrecord files and the supporting "
    "query-docids mapping files.")

flags.DEFINE_string(
    "bert_config_file",
    "./bert_model/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", "../data/output",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_boolean(
    "trec_output", True,
    "Whether to write the predictions to a TREC-formatted 'run' file..")

flags.DEFINE_string(
    "init_checkpoint",
    "./bert_model/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "num_docs", 8,
    "the num of candidate docs")

flags.DEFINE_integer(
    "bert_dim", 768,
    "the num of candidate docs")


flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 2, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 3e-6, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 400000,
                     "Total number of training steps to perform.")

flags.DEFINE_integer(
    "max_dev_examples", None,
    "Maximum number of dev examples to be evaluated. If None, evaluate all "
    "examples in the dev set.")

flags.DEFINE_integer("num_eval_docs", 10,
                     "Number of docs per query in the dev files.")

flags.DEFINE_integer(
    "max_text_examples", None,
    "Maximum number of test examples to be evaluated. If None, evaluate all "
    "examples in the test set.")

flags.DEFINE_integer("num_test_docs", 1000,
                     "Number of docs per query in the dev files.")

flags.DEFINE_integer(
    "num_warmup_steps", 40000,
    "Number of training steps to perform linear learning rate warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

METRICS_MAP = ['MAP', 'RPrec', 'MRR', 'NDCG', 'MRR@10']
FAKE_DOC_ID = "00000000"  # Fake doc id used to fill queries with less than num_eval_docs.


def hinge_loss(similarity_good_tensor, similarity_bad_tensor, margin):
    return tf.maximum(
        0.0,
        tf.add(
            tf.subtract(
                margin,
                similarity_good_tensor
            ),
            similarity_bad_tensor
        )
    )


def cosine_similarity(a, b):
    return tf.math.divide(
        tf.reduce_sum(tf.multiply(a, b), 1),
        tf.multiply(
            tf.sqrt(tf.reduce_sum(tf.square(a), 1)),
            tf.sqrt(tf.reduce_sum(tf.square(b), 1))
        )
    )


def gesd_similarity(a, b):
    a = tf.nn.l2_normalize(a, axis=1)
    b = tf.nn.l2_normalize(b, axis=1)
    euclidean = tf.sqrt(tf.reduce_sum((a - b) ** 2, 1))
    mm = tf.reshape(
        tf.matmul(
            # tf.reshape(a, [-1, 1, tf.shape(a)[1]]),
            tf.expand_dims(a, 1),
            tf.transpose(

                # tf.reshape(b, [-1, 1, tf.shape(a)[1]]),
                tf.expand_dims(b, 1),
                [0, 2, 1]
            )
        ),
        [-1]
    )
    sigmoid_dot = tf.exp(-1 * (mm + 1))
    return 1.0 / (1.0 + euclidean) * 1.0 / (1.0 + sigmoid_dot)


# def get_most_similar_answer(query, bas):
#     queries = tf.expand_dims(query, 0)
#     queries = tf.tile(queries, [FLAGS.num_docs, 1])
#     most_similar_index = tf.argmax(gesd_similarity(queries, bas))
#     return bas[most_similar_index, :]

def get_most_similar_answer(query, bas, bert_batch_num):
    # [-1, max_seq_length]
    queries = tf.tile(query, [1, FLAGS.num_docs])
    queries = tf.reshape(queries, [FLAGS.num_docs * bert_batch_num, -1])

    # [FLAGS.num_docs * bert_batch_num, 1]
    bad_similarity = gesd_similarity(queries, bas)
    # [bert_batch_num, FLAGS.num_docs]
    bad_similarity = tf.reshape(bad_similarity, [bert_batch_num, FLAGS.num_docs])
    # [bert_batch_num, ]
    ind_max = tf.argmax(bad_similarity, axis=1)
    flat_ind_max = ind_max + tf.cast(tf.range(bert_batch_num) * FLAGS.num_docs, tf.int64)
    worst_similar_answer = tf.gather(bas, flat_ind_max)
    return worst_similar_answer


def get_bert_output(bert_config, is_training, features, use_one_hot_embeddings):
    def output(input_ids):
        segment_ids = features["segment_ids"]
        input_mask = features["input_mask"]
        # with tf.variable_scope("bert_out", reuse=reuse):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            # input_mask=input_mask,
            # token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
        r = model.get_pooled_output()
        # return tf.reshape(r, [r.shape[0], -1])
        return r
    return output



def create_model(bert_config, is_training, features, use_one_hot_embeddings):
    doc_ids_list = features["doc_ids_list"]
    qrel_ids = features["qrel_ids"]
    query_ids = features["query_ids"]

    batch_size = FLAGS.train_batch_size
    num_docs = FLAGS.num_docs

    get_output = get_bert_output(bert_config, is_training, features, use_one_hot_embeddings)
    # [batch_size, hidden_size]
    query_bert_outputs = get_output(query_ids)
    # [batch_size, hidden_size]
    good_answers_bert_outputs = get_output(qrel_ids)

    doc_ids_list = tf.reshape(doc_ids_list, [num_docs * batch_size, -1])
    random_idxs = tf.random.uniform([batch_size, ], minval=0, maxval=num_docs, dtype=tf.int64)

    flat_idx = random_idxs + tf.cast(tf.range(batch_size) * num_docs, tf.int64)
    worst_similar_answers = tf.gather(doc_ids_list, flat_idx)
    worst_similar_answers = get_output(worst_similar_answers)

    with tf.variable_scope("loss"):
        query_good_similarity = cosine_similarity(query_bert_outputs, good_answers_bert_outputs)
        query_bad_similarity = cosine_similarity(query_bert_outputs, worst_similar_answers)
        per_example_loss = hinge_loss(query_good_similarity, query_bad_similarity, 0.8)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss)


def create_model1(bert_config, is_training, features, use_one_hot_embeddings):
    doc_ids_list = features["doc_ids_list"]
    qrel_ids = features["qrel_ids"]
    query_ids = features["query_ids"]

    # [batch_size, hidden_size]
    query_bert_outputs = get_bert_output(bert_config, is_training,
                                                features, query_ids, use_one_hot_embeddings,reuse=False)
    # [batch_size, hidden_size]
    good_answers_bert_outputs = get_bert_output(bert_config, is_training,
                                                features, qrel_ids, use_one_hot_embeddings,reuse=True)

    worst_similar_answers_list = []

    # div_num = 32
    div_num = 1
    batch_size = doc_ids_list.shape[0]
    bert_batch_num = batch_size // div_num
    # print("batch_size:{}, bert_batch_num:{}, div_num:{}".format(batch_size, bert_batch_num, div_num))

    for i in range(0, div_num):
        start_idx = bert_batch_num * i
        end_idx = bert_batch_num * (i + 1)
        partial_doc_ids_list = doc_ids_list[start_idx: end_idx, :, :]
        # print("{}, {},partial_doc_ids_list: {}".format(start_idx, end_idx,partial_doc_ids_list))
        partial_doc_ids_list = tf.reshape(partial_doc_ids_list, [bert_batch_num * FLAGS.num_docs, -1])

        partial_bad_answers_bert_outputs = \
            get_bert_output(bert_config, is_training, features, partial_doc_ids_list, use_one_hot_embeddings,reuse=True)

        partial_query_bert_outputs = query_bert_outputs[start_idx: end_idx, :]

        partial_worst_similar_answers = \
            get_most_similar_answer(partial_query_bert_outputs, partial_bad_answers_bert_outputs, bert_batch_num)
        # del partial_bad_answers_bert_outputs
        # len = batch_size
        worst_similar_answers_list.append(partial_worst_similar_answers)

    # [batch_size, hidden_size]
    worst_similar_answers = tf.concat(worst_similar_answers_list, axis=0)

    with tf.variable_scope("loss"):
        query_good_similarity = cosine_similarity(query_bert_outputs, good_answers_bert_outputs)
        query_bad_similarity = cosine_similarity(query_bert_outputs, worst_similar_answers)
        per_example_loss = hinge_loss(query_good_similarity, query_bad_similarity, 0.8)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # create model
        (total_loss, per_example_loss) = create_model(
            bert_config, is_training, features, use_one_hot_embeddings)

        # for n in tf.get_default_graph().get_operations():
        #     print(n.name)
        #
        # print(tvars)
        # exit(0)
        tvars = tf.trainable_variables()

        scaffold_fn = None
        initialized_variable_names = []
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        tf.summary.scalar('Loss', total_loss)
        if mode == tf.estimator.ModeKeys.TRAIN:

            # train_op = optimization.create_optimizer(
            #     total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            # with tf.device('/device:GPU:5'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=tf.train.get_global_step(),
                                          colocate_gradients_with_ops=True)

            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=3)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)

        # elif mode == tf.estimator.ModeKeys.PREDICT:
        #     output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        #         mode=mode,
        #         predictions={
        #             "log_probs": log_probs,
        #             "label_ids": label_ids,
        #             "len_gt_titles": len_gt_titles,
        #         },
        #         scaffold_fn=scaffold_fn)

        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(dataset_path, max_seq_length, num_candidate, is_training,
                     max_eval_examples=None):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""

        batch_size = params["batch_size"]
        output_buffer_size = batch_size * 1000

        def extract_fn(data_record):
            features = {
                "query_ids": tf.io.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "doc_ids": tf.io.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
                "qrel_ids": tf.io.FixedLenSequenceFeature(
                    [], tf.int64, allow_missing=True),
            }
            max_seq_length = FLAGS.max_seq_length
            sample = tf.parse_single_example(data_record, features)

            # question ids
            query_ids = tf.cast(sample["query_ids"], tf.int32)

            #
            doc_ids_list = tf.cast(sample["doc_ids"], tf.int32)
            doc_ids_list = tf.reshape(doc_ids_list, [num_candidate, max_seq_length])

            # right answers ids
            qrel_ids = tf.cast(sample["qrel_ids"], tf.int32)

            # generate type ids for query / right, candidate answers, all zeros, [max_seq_length,]
            segment_ids = tf.zeros([max_seq_length, ], dtype=tf.int32)

            # generate masked id, [max_seq_length, ]
            input_mask = tf.ones_like(segment_ids)

            features = {
                "doc_ids_list": doc_ids_list,
                "qrel_ids": qrel_ids,
                "query_ids": query_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
            }
            return features

        dataset = tf.data.TFRecordDataset(dataset_path)

        dataset = dataset.map(
            extract_fn, num_parallel_calls=4).prefetch(output_buffer_size)

        if is_training:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=1000)
        else:
            if max_eval_examples:
                dataset = dataset.take(max_eval_examples)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes={
                "doc_ids_list": [num_candidate, max_seq_length],
                "query_ids": [50],
                "qrel_ids": [max_seq_length],
                "segment_ids": [max_seq_length],
                "input_mask": [max_seq_length],
            },
            padding_values={
                "doc_ids_list": 0,
                "query_ids": 0,
                "qrel_ids": 0,
                "segment_ids": 0,
                "input_mask": 0,
            },
            drop_remainder=True)

        return dataset

    return input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `FLAGS.do_train` or `FLAGS.do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tpu_cluster_resolver = None
    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            TPU_ADDRESS)

    config = tf.ConfigProto(allow_soft_placement=False)
    # config = tf.ConfigProto(log_device_placement=True)

    # config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True


    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        session_config=config,
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        # tpu_config=tf.contrib.tpu.TPUConfig(
        #     iterations_per_loop=FLAGS.iterations_per_loop,
        #     num_shards=FLAGS.num_tpu_cores,
        #     per_host_input_for_training=is_per_host)
    )


    # session_config = tf.ConfigProto(log_device_placement=True)
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # run_config = tf.estimator.RunConfig().replace(
    #     model_dir=FLAGS.output_dir,
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     session_config=session_config)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)

    # estimator = tf.estimator.Estimator(
    #     model_fn=model_fn,
    #     config=run_config,
    #     params={
    #         'batch_size': FLAGS.train_batch_size
    #     }
    # )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", FLAGS.num_train_steps)
        train_input_fn = input_fn_builder(
            dataset_path=os.path.join(FLAGS.data_dir, "dataset8_test.tf"),
            max_seq_length=FLAGS.max_seq_length,
            num_candidate=FLAGS.num_docs,
            is_training=True)

        estimator.train(input_fn=train_input_fn,
                        max_steps=FLAGS.num_train_steps)
        tf.logging.info("Done Training!")

    if FLAGS.do_eval:
        for set_name in ["dev", "test"]:
            tf.logging.info("***** Running evaluation on the {} set*****".format(
                set_name))
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

            max_eval_examples = None

            if set_name == "dev":
                num_eval_docs = FLAGS.num_dev_docs
                if FLAGS.max_dev_examples:
                    max_eval_examples = FLAGS.max_dev_examples * FLAGS.num_dev_docs

            elif set_name == "test":
                num_eval_docs = FLAGS.num_test_docs
                if FLAGS.max_test_examples:
                    max_eval_examples = FLAGS.max_test_examples * FLAGS.num_test_docs

            eval_input_fn = input_fn_builder(
                dataset_path=os.path.join(FLAGS.data_dir, "dataset_" + set_name + ".tf"),
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                max_eval_examples=max_eval_examples)

            if FLAGS.trec_output:
                trec_file = tf.gfile.Open(
                    os.path.join(
                        FLAGS.output_dir, "bert_predictions_" + set_name + ".run"), "w")
                query_docids_map = []
                docs_per_query = 0  # Counter of docs per query
                with tf.gfile.Open(
                        os.path.join(FLAGS.data_dir, set_name + ".run")) as ref_file:
                    for line in ref_file:
                        query, _, doc_id, _, _, _ = line.strip().split(" ")

                        # We add fake docs so the number of docs per query is always
                        # num_eval_docs.
                        if len(query_docids_map) > 0:
                            if query != last_query:
                                if docs_per_query < num_eval_docs:
                                    fake_pairs = (num_eval_docs - docs_per_query) * [
                                        (last_query, FAKE_DOC_ID)]
                                    query_docids_map.extend(fake_pairs)
                                docs_per_query = 0

                        query_docids_map.append((query, doc_id))
                        last_query = query
                        docs_per_query += 1

            # ***IMPORTANT NOTE***
            # The logging output produced by the feed queues during evaluation is very
            # large (~14M lines for the dev set), which causes the tab to crash if you
            # don't have enough memory on your local machine. We suppress this
            # frequent logging by setting the verbosity to WARN during the evaluation
            # phase.
            tf.logging.set_verbosity(tf.logging.WARN)

            result = estimator.predict(input_fn=eval_input_fn,
                                       yield_single_examples=True)
            start_time = time.time()
            results = []
            all_metrics = np.zeros(len(METRICS_MAP))
            example_idx = 0
            total_count = 0
            for item in result:
                results.append(
                    (item["log_probs"], item["label_ids"], item["len_gt_titles"]))

                if len(results) == num_eval_docs:

                    log_probs, labels, len_gt_titles = zip(*results)
                    log_probs = np.stack(log_probs).reshape(-1, 2)
                    labels = np.stack(labels)
                    len_gt_titles = np.stack(len_gt_titles)
                    assert len(set(list(len_gt_titles))) == 1, (
                        "all ground truth lengths must be the same for a given query.")

                    scores = log_probs[:, 1]
                    pred_docs = scores.argsort()[::-1]

                    gt = set(list(tf.where(labels > 0)[0]))

                    # Metrics like NDCG and MAP require the total number of relevant docs.
                    # The code below adds missing number of relevant docs to gt so the
                    # metrics are the same as if we had used all ground-truths.
                    # The extra_gts have all negative ids so they don't interfere with the
                    # predicted ids, which are all equal or greater than zero.
                    extra_gts = list(-(np.arange(max(0, len_gt_titles[0] - len(gt))) + 1))
                    gt.update(extra_gts)

                    all_metrics += metrics.metrics(
                        gt=gt, pred=pred_docs, metrics_map=METRICS_MAP)

                    if FLAGS.trec_output:
                        start_idx = example_idx * num_eval_docs
                        end_idx = (example_idx + 1) * num_eval_docs
                        queries, doc_ids = zip(*query_docids_map[start_idx:end_idx])
                        assert len(set(queries)) == 1, "Queries must be all the same."
                        query = queries[0]
                        rank = 1
                        for doc_idx in pred_docs:
                            doc_id = doc_ids[doc_idx]
                            score = scores[doc_idx]
                            # Skip fake docs, as they are only used to ensure that each query
                            # has 1000 docs.
                            if doc_id != FAKE_DOC_ID:
                                output_line = " ".join(
                                    (query, "Q0", doc_id, str(rank), str(score), "BERT"))
                                trec_file.write(output_line + "\n")
                                rank += 1

                    example_idx += 1
                    results = []

                total_count += 1

                if total_count % 10000 == 0:
                    tf.logging.warn("Read {} examples in {} secs. Metrics so far:".format(
                        total_count, int(time.time() - start_time)))
                    tf.logging.warn("  ".join(METRICS_MAP))
                    tf.logging.warn(all_metrics / example_idx)

            # Once the feed queues are finished, we can set the verbosity back to
            # INFO.
            tf.logging.set_verbosity(tf.logging.INFO)

            if FLAGS.trec_output:
                trec_file.close()

            all_metrics /= example_idx

            tf.logging.info("Eval {}:".format(set_name))
            tf.logging.info("  ".join(METRICS_MAP))
            tf.logging.info(all_metrics)


if __name__ == "__main__":
    tf.app.run()
