import yaml
import codecs


def main():
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    correctEntities = 0
    responseEntities = 0
    keyEntities = 0
    with codecs.open(config['data_params']['path_test_labeled'], 'r', encoding='utf-8') as fp1, codecs.open(config['data_params']['path_result'], 'r', encoding='utf-8') as fp2:
        for l1, l2 in zip(fp1, fp2):
            if l1 and l1 is not '\n':
                type = l1.strip().split()[1]
                prediction = l2.strip().split()[1]
                if prediction == type and prediction != 'other':
                    correctEntities += 1

                if prediction != 'other':
                    responseEntities += 1

                if type != 'other':
                    keyEntities += 1

    precision = correctEntities / responseEntities
    recall = correctEntities / keyEntities
    f_score = 2 * precision  * recall / (precision + recall)
    print('correct: %s    response: %s    key: %s\n' % (correctEntities, responseEntities, keyEntities))
    print('precision: %s\nrecall: %s\nf_score: %s' % (precision, recall, f_score))


if __name__ == '__main__':
    main()
