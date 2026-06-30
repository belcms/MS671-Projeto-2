def int_to_string(ints, inv_vocab):
    chars = []

    for i in ints:
        c = inv_vocab[i]

        if c == "<pad>":
            continue

        chars.append(c)

    return ''.join(chars)