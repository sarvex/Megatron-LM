# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""QQP dataset."""

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset


LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('QQP', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(f' > Processing {filename} ...')

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split('\t')
                if first:
                    first = False
                    if len(row) == 3:
                        is_test = True
                        print_rank_0(
                            f'   reading {row[0].strip()}, {row[1].strip()}, and {row[2].strip()} columns and setting labels to {self.test_label}'
                        )
                    else:
                        assert len(row) == 6
                        print_rank_0(
                            f'    reading {row[0].strip()}, {row[3].strip()}, {row[4].strip()}, and {row[5].strip()} columns ...'
                        )
                    continue

                if is_test:
                    assert len(row) == 3, f'expected length 3: {row}'
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 6:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[3].strip())
                        text_b = clean_text(row[4].strip())
                        label = int(row[5].strip())
                    else:
                        print_rank_0(f'***WARNING*** index error, skipping: {row}')
                        continue
                    if len(text_a) == 0:
                        print_rank_0(f'***WARNING*** zero length a, skipping: {row}')
                        continue
                    if len(text_b) == 0:
                        print_rank_0(f'***WARNING*** zero length b, skipping: {row}')
                        continue
                assert label in LABELS
                assert uid >= 0

                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    print_rank_0(f'  > processed {total} so far ...')

        print_rank_0(f' >> processed {len(samples)} samples.')
        return samples
