
# with open('data/test.unlabeled', 'r') as fin, open('data/input', 'w') as fout:
#     sentence = []
#     for line in fin:
#         if line != '\n':
#             sentence.append(line.strip())
#         else:
#             fout.write(' '.join(sentence) + '\n')
#             sentence = []
#     if sentence:
#         fout.write(' '.join(sentence) + '\n')
import pickle
import json

with open('models/model_config/mappings.pkl', 'rb') as fp:
    s = pickle.load(fp)
with open('models/model_config/mappings.json', 'w') as fp:
    json.dump(s, fp)
