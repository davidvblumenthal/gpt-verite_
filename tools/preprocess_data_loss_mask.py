# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore

from utils.loss_mask_utils import preprocess_loss_mask
from utils.loss_mask_utils import construct_loss_mask
from utils.loss_mask_utils import split_ids_at_endofsentence_token

from utils.loss_mask_utils import  write_tokenized_text_to_file

from utils.padding_utils import discard_small_samples

import logging

logger = logging.getLogger(__name__)


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            doc_loss_mask = []

            # INSERTED CODE
            if self.args.loss_mask:
                # Temporarliy add special token to tokenizer
                # Get the token id of the <|endofsentence|> token
                eos_id = self.tokenizer.token_to_id('<|endofsentence|>')
                #Encoder.tokenizer.add_special_tokens(["<|endofsentence|>"], special_tokens=True)
                text = preprocess_loss_mask(text)
                
                text_ids = Encoder.tokenizer.tokenize(text)
                text_ids = split_ids_at_endofsentence_token(text_ids, special_token_id=eos_id)
                text_ids, loss_mask = construct_loss_mask(text_ids, self.args.loss_mask_multiple)
            
            else:
                text_ids = Encoder.tokenizer.tokenize(text)
                # Create the loss mask for the samples not using a loss mask by just creating a list of 1s of length text_ids
                loss_mask = [1] * len(text_ids)

            # END INSERTED CODE
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
                doc_loss_mask.append(loss_mask)
                
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
                doc_loss_mask[-1].append(1)

            # Add padding token to the end of the document if necessary
            if self.args.pad_to_max_length:
                max_length = 2048

                # Check if document is longer than max length
                if len(doc_ids[-1]) > max_length:
                    left_over = len(doc_ids[-1]) % max_length
                    num_pad_tokens = max_length - left_over
                    
                    doc_ids[-1].extend([Encoder.tokenizer.pad] * num_pad_tokens)
                    doc_loss_mask[-1].extend([0] * num_pad_tokens)
                
                # Check if document is shorter than max length
                elif len(doc_ids[-1]) < max_length:
                    num_pad_tokens = max_length - len(doc_ids[-1])
                    
                    doc_ids[-1].extend([Encoder.tokenizer.pad] * num_pad_tokens)
                    doc_loss_mask[-1].extend([0] * num_pad_tokens)

                if self.args.discard_samples_smaller is not None:                   
                    # Check if sequence equal or bigger than max_length
                    # equal
                    if len(doc_ids[-1]) == max_length:
                        doc_ids = doc_ids
                        doc_loss_mask = doc_loss_mask
                    # bigger
                    elif len(doc_ids[-1]) > max_length:
                        doc_ids, doc_loss_mask = discard_small_samples(
                        doc_ids=doc_ids[-1],
                        loss_mask=doc_loss_mask[-1],
                        pad_token=Encoder.tokenizer.pad,
                        threshold=self.args.discard_samples_smaller
                        )

                    # Case that should not happend
                    else: 
                        print("Something went wrong! Length should be either equal or bigger than max length. But is is smaller!!!")
                        sys.exit()

                    


                # Sanity check if document is exactly max length or divisible by max length
                assert len(doc_ids[-1]) % max_length == 0, f"Document length {len(doc_ids[-1])} is not equal to or divisible by max length"

                
                    
            
            #ids["text"] = np.vstack([doc_ids, doc_loss_mask])
            ids["text"] = doc_ids
            ids["sc_mask"] = doc_loss_mask

        # DEBUGING
        #write_tokenized_text_to_file(ids, "./final_test.jsonl")
        assert len(doc_loss_mask[0]) == len(doc_ids[0]), f"loss_mask {len(doc_loss_mask[0])} and sentence {len(doc_ids[0])} should have same length"

        

        return ids, len(text)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text", "sc_mask"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "HFGPTVerTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument(
        "--pad-to-max-length",
        action="store_true",
        default=False,
        help="Pad documents to the maximum length of the model.",
    )
    group.add_argument(
        "--discard-samples-smaller",
        type=int,
        help="Sample will be discarded if it contains less than 50 tokens"
    )

    group.add_argument(
        "--loss-mask-multiple",
        type=float,
        default=2,
        help="The multiple to use for the loss mask. Default: 1.25"
    )
    group.add_argument(
        "--loss-mask",
        action="store_true",
        default=False,
        help="Does the dataset to be tokenized implement the loss mask or not?"
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = get_args()

    logger.info(f"=======Running Tokenization with arguments: {args} =========")

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(args.input.split(","), semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}.bin".format(
            args.output_prefix, key
        )
        output_idx_files[key] = "{}_{}.idx".format(
            args.output_prefix, key
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])


"""

python preprocess_data_loss_mask.py \
            --input /home/kit/stud/ukmwn/master_thesis/data/Wikipedia/Wikipedia_sample_en.jsonl \
            --output-prefix ../../data/padding/verzweifelt \
            --vocab ../../data/les_faits/tokenizer/gpt-ver-tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFGPTVerTokenizer \
            --loss-mask-multiple 2 \
            --loss-mask \
            --pad-to-max-length \
            --append-eod

"""


if __name__ == "__main__":
    main()
