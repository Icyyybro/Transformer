from make_model import make_model


source_vocab = 11
target_vocab = 11
N = 6


if __name__ == '__main__':
    res = make_model(source_vocab, target_vocab, N)
    print(res)