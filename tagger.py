#!/usr/bin/env python

from __future__ import print_function
from builtins import zip
from builtins import range
import os
import time
import codecs
import optparse
import yaml
from loader import prepare_sentence, prepare_sentence_
from utils import create_input_batch, zero_digits
from model import Model
import tensorflow as tf

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="models/lower=False,zeros=False,char_dim=30,char_lstm_dim=30,char_bidirect=True,word_dim=200,word_lstm_dim=200,word_bidirect=True,crf=True,dropout=0.1,lr_method=adam,lr_rate=0.005,clip_norm=0.0,is_train=1,batch_size=32",
    help="Model location"
)
optparser.add_option(
    "-s", "--saver", default="models/saver",
    help="tf checkpoint location"
)
optparser.add_option(
    "-i", "--input", default="data/input",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="data/output",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="\t",
    help="Delimiter to separate words from their tags"
)

opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isdir(opts.saver)
assert os.path.isfile(opts.input)

# load config
with open('./config.yml') as file_config:
    config = yaml.load(file_config)

# Load existing model
print("Loading model...")
model = Model(config, model_path=opts.model)
parameters = model.parameters
parameters['is_train'] = 0
parameters['dropout'] = 0
batch_size = parameters['batch_size']
# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in list(x.items())}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]
tag_count = len(tag_to_id)
# Load the model
cost, f_eval, _ = model.build(config=config, **parameters)

f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()
saver = tf.train.Saver()
print('Tagging...')
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(opts.saver)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    test_data = []
    word_data = []
    with codecs.open(opts.input, 'r', 'utf-8') as f_input:
        for line in f_input:
            words = line.rstrip().split()
            if line:
                # Lowercase sentence
                if parameters['lower']:
                    line = line.lower()
                # Replace all digits with zeros
                if parameters['zeros']:
                    line = zero_digits(line)
                # Prepare input
                if parameters['char_dim']:
                    sentence = prepare_sentence(words, word_to_id, char_to_id,
                                                lower=parameters['lower'])
                else:
                    sentence = prepare_sentence_(words, word_to_id,lower=parameters['lower'])
                test_data.append(sentence)
                word_data.append(words)
            else:
                continue
    count = 0
    assert len(test_data) == len(word_data)
    while count < len(test_data):
        batch_data = []
        batch_words = []
        for i in range(batch_size):
            index = i + count
            if index >= len(test_data):
                break
            data = test_data[index]
            batch_data.append(test_data[index])
            batch_words.append(word_data[index])
        if len(batch_data) <= 0:
            break
        # try:
        input_ = create_input_batch(batch_data, parameters)
        # except ValueError as e:
        #     print('fail when batch number %d' % count)
        #     exit(1)
        feed_dict_ = {}
        if parameters['char_dim']:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
            feed_dict_[model.char_for_ids] = input_[2]
            feed_dict_[model.char_rev_ids] = input_[3]
            feed_dict_[model.char_pos_ids] = input_[4]
        else:
            feed_dict_[model.word_ids] = input_[0]
            feed_dict_[model.word_pos_ids] = input_[1]
        f_scores = sess.run(f_eval, feed_dict=feed_dict_)
        # Decoding
        if parameters['crf']:
            for x in range(len(batch_data)):
                f_score = f_scores[x]
                word_pos = input_[1][x] + 2
                y_pred = f_score[1:word_pos]
                words = batch_words[x]
                y_preds = [model.id_to_tag[pred] for pred in y_pred]
                assert len(words) == len(y_preds)
                # Write tags
                # f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))
                f_output.write('%s\n\n' % '\n'.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))

        else:
            f_score = f_scores.argmax(axis=-1)
            for x in range(len(batch_data)):
                word_pos = input_[1][x] + 1
                y_pred = f_score[x][0:word_pos]
                words = batch_words[x]
                y_preds = [model.id_to_tag[pred] for pred in y_pred]
                assert len(words) == len(y_preds)
                # Write tags
                # f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))
                f_output.write('%s\n\n' % '\n'.join('%s%s%s' % (w, opts.delimiter, y) for w, y in zip(words, y_preds)))

        count += len(batch_data)
print('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))
f_output.close()
