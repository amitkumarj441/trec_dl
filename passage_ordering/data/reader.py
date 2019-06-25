import gzip

from passage_ordering.data.cbor_read_data import iter_pages


class ArchiveReader(object):
    def __init__(self, archive_path, lowercased=False, logger=None):
        self.archive_path = archive_path
        self.lowercased = lowercased
        self.logger = logger

    def read(self, is_gzip=False):
        raise NotImplementedError()


class PageReader(ArchiveReader):
    # get all the paragraphs from a heading
    def get_text_list(self, pre_head, line):
        head = pre_head + ' / ' + line.heading
        result = []
        for l in line.get_text().split('\n'):
            result.append([head, l, 1])
        return result

    def read_cbor(self, file_path, is_gzip=False):
        result = []

        if is_gzip:
            f = gzip.open(file_path, 'rb')
        else:
            f = open(file_path, 'rb')

        try:
            for p in iter_pages(f):
                prev_head = p.page_name

                if len(p.outline()) > 0:
                    for index, line in enumerate(p.outline()):
                        result.extend(self.get_text_list(prev_head, line))
                        prev_subhead = prev_head + ' / ' + line.heading

                        # sub_heading
                        for child in line.child_sections:
                            result.extend(self.get_text_list(prev_subhead, child))
        finally:
            f.close()
        return result






