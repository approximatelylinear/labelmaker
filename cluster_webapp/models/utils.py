
#   Stdlib
import random


def sample(df, n=50):
    """
    Samples a dataframe

    :param pd.DataFrame df: Dataframe to sample
    :param int n: Number of items (defaults to 50)

    """
    size = len(df)
    if n < size:
        idxs = random.sample(xrange(size), n)
        samples= df.iloc[idxs]
    else:
        samples = df
    return samples


def ravel_labels(df):
    """
    (Un)ravels a dataframe: Creates a new row for each token in the column 'labels' for each row.

    :param pd.DataFrame df: Dataframe to unravel.

    :return: The unraveled dataframe
    :rtype: pd.DataFrame

    """
    if any(df.label.map(lambda x: isinstance(x, basestring))):
        df.label = df.label.map(lambda x: x.split(','))
    rows = []
    for k, row in df.iterrows():
        d = row.to_dict()
        labels = d['label']
        #   Normalize and de-dup the labels.
        labels = [t.lower().strip() for t in labels]
        labels = set(labels)
        #   Add a new row for each label.
        for label in labels:
            new_d = dict(d)
            new_d['label'] = label
            rows.append(new_d)
    new_df = DataFrame(rows)
    return new_df


def ravel_label_row(d):
    """
    Creates a new row for each label in the data.

    :param dict d: Dictionary representing a DataFrame row.

    :return: Generator of new entries
    :rtype: Generator

    """
    labels = d.pop('label', []) or d.pop('labels', [])
    #   Normalize and de-dup the labels.
    labels = [t.lower().strip() for t in labels]
    labels = set(labels)
    #   Add a new row for each label.
    for label in labels:
        new_d = dict(d)
        new_d['label'] = label
        yield new_d


def sql_2_dict(obj):
    """
    Converts a sqlalchemy ORM object to a dictionary.
    """
    return dict([
        (k, getattr(obj, k))
            for k in dir(obj)
                if k == '_id' or not (
                    k.startswith('_') or
                    k in ['metadata'] or
                    callable(getattr(obj, k))
                )
    ])
