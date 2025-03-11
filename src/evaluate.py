from hiclass.metrics import precision, recall, f1

def accuracy_class(y_true, y_pred, level):

    total, hits = defaultdict(int), defaultdict(int)

    for t, p in zip(y_true, y_pred):

        total[t[level]] += 1
        if t[level] == p[level]:
            hits[t[level]] += 1

    return {classe: hits[classe] / total[classe] for classe in total}

def accuracy_unit(units, true, pred, level):

    acc = []
    for unit in set(units):
        true_vals = [t[level] for u, t in zip(units, true) if u == unit]
        pred_vals = [p[level] for u, p in zip(units, pred) if u == unit]
        acc.append((unit, true_vals, pred_vals))

    return acc

def accuracy_level(y_true, y_pred, level):
    acc = [(1 if true[level] == '' or true[level] == pred[level] else 0) for true, pred in zip(y_true, y_pred)]
    return sum(acc)/len(acc)

def flatly(y_true, y_pred):
    return {'Level ' + str(level) : accuracy_level(y_true, y_pred, level) for level in range(4)}

def hierarchy(y_true, y_pred, type='micro'):
    return {'F1-score': f1(y_true, y_pred, type),
            'Precision': precision(y_true, y_pred, type),
            'Recall': recall(y_true, y_pred, type)}

def performance(y_true, y_pred):
    return hierarchy(y_true, y_pred) | flatly(y_true, y_pred)