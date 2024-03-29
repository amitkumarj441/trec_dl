import os
from collections import OrderedDict

from passage_ordering.data.reader import PageReader


def _get_text_item(text, id):
    question_tokens = [Token(t) for t in text.split()]
    question_sentence = Sentence(' '.join([t.text for t in question_tokens]), question_tokens)
    ti = TextItem(question_sentence.text, [question_sentence])
    ti.metadata['id'] = id
    return ti


class WikiQAReader(PageReader):
    def read_split(self, name):
        wikiqa_path = os.path.join(self.archive_path, name)
        data = self.read_cbor(wikiqa_path)
        questions_answers = OrderedDict()
        for i, line in enumerate(data):
            question_line = line[0]
            answer_line = line[1]
            label_line = line[2]

            if question_line not in questions_answers:
                questions_answers[question_line] = []
            questions_answers[question_line].append((answer_line, label_line))

        datapoints = []
        # 所有的answer，每个item是一个answer
        split_answers = []
        for i, (question, answers) in enumerate(questions_answers.items()):
            question_item = _get_text_item(question, 'question-{}-{}'.format(name, i))
            ground_truth = []

            # todo discuss how to add candidate answers here
            candidate_answers = []
            for j, (answer, label) in enumerate(answers):
                answer_item = _get_text_item(answer, 'answer-{}-{}'.format(name, j))
                if label == '1':
                    ground_truth.append(answer_item)
                candidate_answers.append(answer_item)

            split_answers += candidate_answers
            if len(ground_truth) > 0:
                datapoints.append(QAPool(question_item, candidate_answers, ground_truth))

        return Data(name, datapoints, split_answers)

    def read(self):
        train = self.read_split("train")
        valid = self.read_split("dev")
        test = self.read_split("test")

        #todo 所有answer question 都合并？？
        questions = [qa.question for qa in (train.qa + valid.qa + test.qa)]
        answers = train.answers + valid.answers + test.answers

        #todo 为啥test要转换成list
        return Archive(train, valid, [test], questions, answers)


class WikiQAData(QAData):
    def _get_reader(self):
        return WikiQAReader(self.config['wikiqa'], self.lowercased, self.logger)


component = WikiQAData
