from typing import Dict, List, Tuple, Set, Text
import textdistance
import string
import re
from tqdm import tqdm
from copy import copy
import random
from tqdm.auto import tqdm
import pandas as pd


def initial_char(text):
    return [s[0] for s in text.split()]


def hamming_search(query: str, text: str, spans: List, ent="ORG"):
    """Search similarities in entities, based on hamming distance between two
    entities, with more bias toward first and intitial charater simialrites

    Args:
    query(str): the entity to search it's similar in text
    text(str): text that query meant to search inside
    ent(str): NER tag to match with query
    """
    query = query.lower().translate(str.maketrans("", "", string.punctuation))
    ents = [x["text"] for x in spans if x["label"] == ent]

    search = set(filter(None, ents))
    main_ents = list(search)
    clean_search = filter(
        None,
        [
            t.lower().translate(str.maketrans("", "", string.punctuation))
            for t in search
        ],
    )
    results = [
        textdistance.hamming.normalized_similarity(query.split()[0], s.split()[0])
        + textdistance.hamming.normalized_similarity(query, s)
        + textdistance.hamming.normalized_similarity(query, initial_char(s))
        + textdistance.hamming.normalized_similarity(initial_char(query), s)
        for s in clean_search
    ]
    if len(results) == 0:
        return None, 0.0
    if max(results) <= 0.6:
        return None, max(results)
    index = results.index(max(results))
    e = main_ents[index]
    return e, max(results)


def word_search(word: str, text: str) -> List[Tuple[int]]:
    """Extarct start&end char span for specific word in a text.

    Args:
        word (str): query word to search
        text (str): Text to be searched

    Returns:
        List with spans tuple(start,end)

    """
    return (
        [(ele.start(), ele.end()) for ele in re.finditer(word.lower(), text.lower())]
        if word is not None
        else []
    )


def Intersection(lst1: List, lst2: List) -> Set:
    """Intersection between two lists

    Args:
        lst1 (List)
        lst2 (List)

    Returns:
        Set of intercetion
    """
    return set(lst1).intersection(lst2)


def create_re_data(
    dataframe: pd.DataFrame,
    text: Text,
    ent1: Text,
    ent2: Text,
    label: Text,
    random_choice: bool = False,
) -> pd.DataFrame:
    """Create entity relation extraction model by insert entity tags around each entity token given in dataframe rows

    Returns:
        _type_: _description_
    """
    chars = string.ascii_letters
    sentences = []
    labels = []
    low_score_sample = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        relation = row[label]
        if row["Label"] == 0:
            relation = "other"
        sentence, e1, e2 = row[text], row[ent1], row[ent2]
        res1 = word_search(e1, sentence)

        if len(res1) == 0:
            # print(f'did not find Filer : <{e1}>\n>>>Search in Orgs')
            b1 = copy(e1)
            e1, score1 = hamming_search(e1, sentence, row["spans"])
            res1 = word_search(e1, sentence)
            # print(f'>>>Found <{e1}> as most similar with score {score1}')
            if score1 < 1.0 and score1 > 0.6:
                low_score_sample.append(
                    {"Sentence": sentence, "Query": b1, "Result": e1, "Score": score1}
                )

        res2 = word_search(e2, sentence)
        if len(res2) == 0:
            # print(f'did not find Company : <{e2}>\n>>>Search in Orgs')
            b2 = copy(e2)
            e2, score2 = hamming_search(e2, sentence, row["spans"])
            res2 = word_search(e2, sentence)
            if score2 < 1.0 and score2 > 6.0:
                low_score_sample.append(
                    {"Sentence": sentence, "Query": b2, "Result": e2, "Score": score2}
                )

            # print(f'>>>Found <{e2}> as most similar with score {score2} ')

        if e1 == e2:
            continue

        if len(res1) == 0 or len(res2) == 0:
            continue

        if random_choice:
            res1 = [random.choice(res1)]
            res2 = [random.choice(res2)]

        for j, r1 in enumerate(res1):
            s = sentence[: r1[0]] + "[E1] " + sentence[r1[0] :]
            s = s[0 : (r1[1] + 5)] + " [/E1]" + s[(r1[1] + 5) :]

            res1 = word_search(e1, s)
            r1 = res1[j]
            res2 = word_search(e2, s)
            for i, r in enumerate(res2):
                if i == 1:
                    break
                intersec = Intersection(
                    list(range(r[0], r[1])), list(range(r1[0], r1[1]))
                )
                if len(intersec) > 0:
                    continue

                if r[0] < r1[0]:
                    r2 = r[0], r[1]
                else:
                    r2 = r[0], r[1]

                intersec = Intersection(
                    list(range(r2[0], r2[1])), list(range(r1[0], r1[1]))
                )
                if len(intersec) > 0:
                    continue
                out = s[: r2[0]] + "[E2] " + s[r2[0] :]
                out = out[0 : (r2[1] + 5)] + " [/E2]" + out[(r2[1] + 5) :]
                sentences.append(out)
                labels.append(relation)
                # print(out)

    return pd.DataFrame({"sents": sentences, "relations": labels})


def stratify_split(data, stratify_by, frac=0.2, random_state=200):
    """Split data into train and test with stratifing column"""
    label_groups = data.groupby(stratify_by)
    train = pd.DataFrame()
    test = pd.DataFrame()
    for i, group in label_groups:
        g_test = group.sample(frac=frac, random_state=random_state)
        g_train = group.drop(g_test.index)
        test = pd.concat([test, g_test], axis=0)
        train = pd.concat([train, g_train], axis=0)

    return train.reset_index(drop=True), test.reset_index(drop=True)
