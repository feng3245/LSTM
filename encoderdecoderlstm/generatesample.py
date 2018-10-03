from random import seed
from random import randint
from math import ceil, log10
from numpy import array, argmax


def to_expression_str(numbers, ops):
    expr = str(numbers[0])
    for n, o in zip(numbers[1:], ops):
        expr = expr +str(o)+ str(n)
    return expr

def calculate_exp(numbers, ops):
    return round(eval(to_expression_str(numbers, ops)))
    
    
#To handle variable equation one must train for it
def random_sum_pairs(n_examples, n_numbers, largest, alphabet):
    X, y = list(), list()
    Nops = list()
    for _ in range(n_examples):
        terms = randint(2,n_numbers)
        in_pattern = [randint(1, largest) for _ in range(terms)]
        ops = generate_random_operations(terms - 1, alphabet)
        Nops.append(ops)
        out_pattern = calculate_exp(in_pattern, ops)
        X.append(in_pattern)
        y.append(out_pattern)
    return X, y, Nops


def to_string(X, y, n_numbers, largest, Nops):
    max_length = int(n_numbers * ceil(log10(largest+1)) + n_numbers - 1)
    Xstr = list()
    for pattern, ops in zip(X, Nops):
        strp = to_expression_str(pattern, ops)
        strp = ''.join([' ' for _ in range(max_length-len(strp))]) + strp
        Xstr.append(strp)
    max_length = int(ceil(log10(pow(largest+1, n_numbers))))
    ystr = list()
    for pattern in y:
        strp = str(pattern)
        strp = ''.join([' ' for _ in range(max_length - len(strp))]) + strp
        ystr.append(strp)
    return Xstr, ystr

def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]
        yenc.append(integer_encoded)
    return Xenc, yenc

def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)
    yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        yenc.append(pattern)
    return Xenc, yenc

def generate_random_operations(n_ops, alphabet):
    ops = []
    for _ in range(n_ops):
        ops.append(alphabet[randint(10,13)])
    return ops


def generate_data(n_samples, n_numbers, largest, alphabet):
    X, y, Nops = random_sum_pairs(n_samples, n_numbers, largest, alphabet)
    
    X, y = to_string(X, y, n_numbers, largest, Nops)

    X, y = integer_encode(X, y, alphabet)

    X, y = one_hot_encode(X, y, len(alphabet))

    X, y = array(X), array(y)
    return X, y

def invert(seq, alphabet):
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    strings = list()
    for pattern in seq:
        string = int_to_char[argmax(pattern)]
        strings.append(string)
    return ''.join(strings)

