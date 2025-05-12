# SPDX-FileCopyrightText: 2028-present Marceau <git@marceau-h.fr>
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from io import StringIO
from re import Pattern, compile, escape
from pathlib import Path
from typing import Tuple, Optional, List, Union
from collections.abc import Collection
from unicodedata import normalize

import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_DEV_TEST_SPLIT = (.7, .15, .15)
SHUFFLE = True

SOS_ID = 0
SOS_TOKEN = 'SOS'
EOS_ID = 1
EOS_TOKEN = 'EOS'
PAD_ID = 2
PAD_TOKEN = 'PAD'

assert .999 < sum(TRAIN_DEV_TEST_SPLIT) < 1.001, f"TRAIN_DEV_TEST_SPLIT must sum to 1, got {sum(TRAIN_DEV_TEST_SPLIT)}"
TRAIN_PART = TRAIN_DEV_TEST_SPLIT[0]
DEV_PART = TRAIN_DEV_TEST_SPLIT[1]
TEST_PART = TRAIN_DEV_TEST_SPLIT[2]
DEV_TEST_RATIO = DEV_PART / (DEV_PART + TEST_PART)

class Language:
    # Bring the constants as class attrs
    SOS_ID = SOS_ID
    SOS_TOKEN = SOS_TOKEN
    EOS_ID = EOS_ID
    EOS_TOKEN = EOS_TOKEN
    PAD_ID = PAD_ID
    PAD_TOKEN = PAD_TOKEN

    def __init__(self, name:str, sep: Optional[Union[str, List[str]]] = None) -> None:
        self.name: str = name
        self.token2index: dict[str, int] = {
            self.SOS_TOKEN: self.SOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.PAD_TOKEN: self.PAD_ID
        }
        self.token2count: dict[str, int] = {}
        # self.index2token = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.index2token: dict[int, str] = {
            self.SOS_ID: self.SOS_TOKEN,
            self.EOS_ID: self.EOS_TOKEN,
            self.PAD_ID: self.PAD_TOKEN
        }
        self.n_tokens: int = 3
        self.max_length: int = 0
        self.sep: Optional[Union[str, List[str]]] = sep
        self.re_sep : Optional[str] = None
        self.re_sep_compiled : Optional[Pattern] = None

    @staticmethod
    def normalize(s: str) -> str:
        """
        Normalize a string to NFKC form (the most composed form)
        Prevents from getting two codes for the same character, i.e 'é' (U+00E9) and 'é' (U+0065 U+0301)
        :param s: string to normalize
        :return: normalized string
        """
        return normalize('NFKC', s)

    def sent_iter(self, sentence: str) -> Collection[str]:
        """
        Function to iterate over the tokens of a sentence
        :param sentence: sentence to iterate over
        :return: iterator over the tokens of the sentence, split by the good separator
        """
        if self.sep is None:
            return sentence

        if self.re_sep is None:
            if self.re_sep is not None:
                self.re_sep_compiled = compile(self.re_sep)
                return self.re_sep_compiled.split(sentence)
            elif isinstance(self.sep, list):
                self.re_sep = '|'.join(f"(?:{escape(s)})" for s in self.sep)
                self.re_sep_compiled = compile(self.re_sep)
                return self.re_sep_compiled.split(sentence)
            elif isinstance(self.sep, str):
                return sentence.split(self.sep)
            else:
                raise ValueError("sep must be a string or a list of strings")
        elif isinstance(self.re_sep_compiled, Pattern):
            return self.re_sep_compiled.split(sentence)
        else:
            raise ValueError("re_sep must be a Pattern object (if defined)")

    def sent_uniter(self, sentence: Collection[str]) -> str:
        """
        Function to unite the tokens of a sentence
        :param sentence: sentence to unite
        :return: united sentence
        """
        if self.sep is None:
            return "".join(sentence)

        if isinstance(self.sep, str):
            return self.sep.join(sentence)

        if isinstance(self.sep, list):
            return self.sep[0].join(sentence)

        raise ValueError("sep must be a string or a list of strings, can't concatenate back the sentence with the given sep")

    def sent_len(self, sentence: str) -> int:
        """
        Function to get the length of a sentence
        :param sentence: sentence to get the length of
        :return: length of the sentence
        """
        return len(self.sent_iter(sentence))

    def add_sentence(self, sentence: str) -> None:
        """
        Function to add a sentence to the language
        :param sentence: sentence to add
        :return: None
        """
        sentence = self.normalize(sentence)
        iterator_ = self.sent_iter(sentence)

        for token in iterator_:
            self.add_token(token)

        self.max_length = max(self.max_length, len(iterator_))

    def add_token(self, token: str) -> None:
        """
        Function to add a token to the language (if it is not already in it)
        :param token: token to add
        :return: None
        """
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    def indices_from_sentence(self, sentence: str) -> List[int]:
        """
        Function to get the indices from a sentence
        :param sentence: sentence to get the indices from
        :return: list of indices
        """
        return [self.SOS_ID] + [self.token2index[token] for token in self.sent_iter(sentence)] + [self.EOS_ID] + [self.PAD_ID] * (self.max_length - self.sent_len(sentence))

    def sentence_from_indices(self, indices: List[int]) -> str:
        """
        Function to get the sentence from the indices
        :param indices: list of indices to get the sentence from
        :return: sentence
        """
        return self.sent_uniter([self.index2token[index] for index in indices if index not in [self.SOS_ID, self.EOS_ID, self.PAD_ID]])

    @classmethod
    def read_data_from_txt(
            cls,
            data_path: Union[str, Path],
            max_length=75,
            l1_sep: Optional[str] = None,
            l2_sep: Optional[str] = " | ",
            pairs_sep: str = "\t",
            instance_sep: str = "\n",
            from_lang: Optional[Path] = None,
    ) -> Tuple[np.array, np.array, "Language", "Language"]:
        """
        Function to read data from a txt file
        :param data_path: path to the data file
        :param max_length: maximum length of a sentence
        :param l1_sep: The separator for the first language
        :param l2_sep: The separator for the second language
        :param pairs_sep: The separator between the pairs
        :param instance_sep: The separator between the instances
        :return: Tuple of X, y, Language object for the input language, Language object for the output language
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        elif not isinstance(data_path, Path):
            raise ValueError("data_path must be a string or a Path object")

        assert data_path.exists(), f"Data path {data_path} does not exist"

        with data_path.open("r") as f:
            iterable = f if instance_sep == "\n" else f.read().split(instance_sep)
            pairs = [cls.normalize(elem.strip()).split(pairs_sep) for elem in iterable if elem.strip()]

        pairs = [
            (
                p0, p1
            )
            for p0, p1 in pairs
            if 0 < len(p0) <= max_length
            and 0 < len(p1) <= max_length
        ]

        if from_lang is None:
            l1 = cls('1', sep=l1_sep)
            l2 = cls('2', sep=l2_sep)
        else:
            with open(from_lang, 'r') as f:
                lang = json.load(f)

            l1 = cls('1')
            l2 = cls('2')

            l1.restore_lang(lang['1'])
            l2.restore_lang(lang['2'])

            l1.sep = l1_sep
            l2.sep = l2_sep


        for pair in pairs:
            l1.add_sentence(pair[0])
            l2.add_sentence(pair[1])

        X, y = zip(
            *[
                (
                    [cls.SOS_ID] +
                    [
                        l1.token2index[token] for token in l1.sent_iter(pair0)
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l1.max_length - l1.sent_len(pair0)),
                    [cls.SOS_ID] +
                    [
                        l2.token2index[token] for token in l2.sent_iter(pair1)
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l2.max_length - l2.sent_len(pair1))
                )
                for pair0, pair1 in pairs
                # if (len_pair0 := len(pair[0])) <= max_length
                # and (len_pair1 := len(pair[1])) <= max_length
            ]
        )

        print(len(X), len(y))
        print(X[0], y[0])
        print(len(X[0]), len(y[0]))

        X = np.array(X)
        y = np.array(y)
        return X, y, l1, l2

    @classmethod
    def read_data_from_json(
            cls,
            data_path: Union[str, Path],
            max_length=1000,
            l1_sep = None,
            l2_sep = None,
    ) -> Tuple[np.array, np.array, "Language", "Language"]:
        """
        Function to read data from a json file
        :param data_path: The path to the json file
        :param max_length: maximum length of a sentence
        :param l1_sep: The separator for the first language
        :param l2_sep: The separator for the second language
        :param reverse: Whether to reverse the pairs or not (i.e. switch the input and output) (default: True)
        :return: Tuple of X, y, Language object for the input language, Language object for the output language
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)
        elif not isinstance(data_path, Path):
            raise ValueError("data_path must be a string or a Path object")

        assert data_path.exists(), f"Data path {data_path} does not exist"

        with data_path.open("r") as f:
            pairs = json.load(f)

        l1 = cls('1', sep=l1_sep)
        l2 = cls('2', sep=l2_sep)

        if isinstance(pairs, dict):
            pairs = [
                (
                    e.strip(), k.strip()
                )
                for k, v in pairs.items()
                for e in v
                if 0 < l2.sent_len(k) <= max_length
                and 0 < l1.sent_len(e) <= max_length
            ]
        elif isinstance(pairs, list):
            pairs = [
                (
                    e["input"].strip(), e["output"].strip()
                )
                for e in pairs
                if 0 < l1.sent_len(e["input"]) <= max_length
                and 0 < l2.sent_len(e["output"]) <= max_length
            ]


        pairs = [
            (
                e_1, e_2
            )
            for e_1, e_2 in pairs
            if e_1 and e_2
        ]


        lens = [l1.sent_len(pair[0]) for pair in pairs]
        print(len(lens), min(lens), max(lens), sum(lens) / len(lens))
        lens2 = [l2.sent_len(pair[1]) for pair in pairs]
        print(len(lens2), min(lens2), max(lens2), sum(lens2) / len(lens2))


        for pair in pairs:
            l1.add_sentence(pair[0])
            l2.add_sentence(pair[1])

        print("L1", l1.token2index)
        print("L2", l2.token2index)

        X, y = zip(
            *[
                (
                    [cls.SOS_ID] +
                    [
                        l1.token2index[token] for token in l1.sent_iter(pair0)
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l1.max_length - l1.sent_len(pair0)),
                    [cls.SOS_ID] +
                    [
                        l2.token2index[token] for token in l2.sent_iter(pair1)
                    ] + [cls.EOS_ID] + [cls.PAD_ID] * (l2.max_length - l2.sent_len(pair1))
                )
                for pair0, pair1 in pairs
                # if (len_pair0 := len(pair[0])) <= max_length
                # and (len_pair1 := len(pair[1])) <= max_length
            ]
        )

        print(len(X), len(y))
        print(X[0], y[0])
        print(len(X[0]), len(y[0]))

        X = np.array(X)
        y = np.array(y)
        return X, y, l1, l2


    @classmethod
    def load_data(
            cls,
            X_path: Union[str, Path],
            y_path: Union[str, Path],
            lang_path: Union[str, Path],
    ) -> Tuple[np.array, np.array, "Language", "Language"]:
        """
        Loads back the data from the files
        :param X_path: The path to the X data
        :param y_path: The path to the y data
        :param lang_path: The path to the language file (containing the two languages for X and y)
        :return:
        """
        if isinstance(X_path, str):
            X_path = Path(X_path)
        elif not isinstance(X_path, Path):
            raise ValueError("X_path must be a string or a Path object")

        if isinstance(y_path, str):
            y_path = Path(y_path)
        elif not isinstance(y_path, Path):
            raise ValueError("y_path must be a string or a Path object")

        if isinstance(lang_path, str):
            lang_path = Path(lang_path)
        elif not isinstance(lang_path, Path):
            raise ValueError("lang_path must be a string or a Path object")

        assert X_path.exists(), f"X path {X_path} does not exist"
        assert y_path.exists(), f"y path {y_path} does not exist"
        assert lang_path.exists(), f"Language path {lang_path} does not exist"

        X, y = np.load(X_path), np.load(y_path)

        with open(lang_path, 'r') as f:
            lang = json.load(f)

        l1 = cls('1')
        l2 = cls('2')

        l1.restore_lang(lang['1'])
        l2.restore_lang(lang['2'])

        return X, y, l1, l2

    def reintify_lang(self) -> None:
        """
        Make lang int again
        Reconverts the indices from strs to ints
        :return:
        """
        self.token2index = {k: int(v) for k, v in self.token2index.items()}
        self.index2token = {int(k): v for k, v in self.index2token.items()}

    def restore_lang(self, lang: dict) -> None:
        """
        Restore the language from a dict
        :param self: The language object in which to restore the language
        :param lang: dict to restore the language from
        :return: None
        """
        self.__dict__.update(lang)
        self.reintify_lang()

    @staticmethod
    def clear_pattern_field_only(obj: object) -> None:
        """
        Function to avoid errors when trying to serialize a Pattern object, which is not serializable
        It will clear this field by setting it to None but will raise a TypeError if the object is not a Pattern object to avoid unwanted behavior
        :param obj: object to clear the pattern field
        :return: None
        """
        if isinstance(obj, Pattern):
            return None
        else:
            raise TypeError

    @classmethod
    def save_data(
            cls,
            X: np.array,
            y: np.array,
            l1: "Language",
            l2: "Language",
            lang_path: Union[str, Path],
            overwrite: bool = False,
    ) -> None:
        """
        Save the data to the files
        :param X: The numpy array of the input data
        :param y: The numpy array of the output data
        :param l1: The Language object for the input language
        :param l2: The Language object for the output language
        :param X_path: The path to save the X data
        :param y_path: The path to save the y data
        :param lang_path: The path to save the language file
        :return: None
        """
        assert isinstance(lang_path, Path), f"lang_path must be a string or a Path object"
        try:
            lang_path.mkdir(parents=True, exist_ok=overwrite)
        except FileExistsError:
            if not overwrite:
                raise FileExistsError(f"Directory {lang_path} already exists, please provide a different path or set overwrite to True")

        np.save(lang_path / 'X.npy', X)
        np.save(lang_path / 'y.npy', y)

        with open(lang_path / 'lang.json', 'w') as f:
            json.dump({'1': l1.__dict__, '2': l2.__dict__}, f, ensure_ascii=False, indent=4, default=cls.clear_pattern_field_only)

def read_data(lang_path: Union[str, Path]) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, "Language", "Language"]:
    """
    Reads the data from the files
    The data is split into X and y data, with a determined random state and test size (see constants) to be reproducible
    And the languages are loaded from the language file
    :param x_path: The path to the X data
    :param y_path: The path to the y data
    :param lang_path: The path to the language file (containing the two languages for X and y)
    :return: Tuple of X_train, X_dev, X_test, y_train, y_dev, y_test, Language object for the input language, Language object for the output language
    """
    assert .999 < sum(TRAIN_DEV_TEST_SPLIT) < 1.001, f"TRAIN_DEV_TEST_SPLIT must sum to 1, got {sum(TRAIN_DEV_TEST_SPLIT)}"

    assert isinstance(lang_path, Path), f"lang_path must be a string or a Path object"
    assert lang_path.exists(), f"Language path {lang_path} does not exist, please provide a valid path"

    x_path = lang_path / 'X.npy'
    y_path = lang_path / 'y.npy'
    l_path = lang_path / 'lang.json'

    assert x_path.exists(), f"X path {x_path} does not exist"
    assert y_path.exists(), f"y path {y_path} does not exist"
    assert l_path.exists(), f"Language path {l_path} does not exist"

    X, y, lang_input, lang_output = Language.load_data(x_path, y_path, l_path)
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=1-TRAIN_PART, random_state=RANDOM_STATE, shuffle=SHUFFLE)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=DEV_TEST_RATIO, random_state=RANDOM_STATE, shuffle=SHUFFLE)


    return X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output


def extract_test_data(
        x_path: Union[str, Path] = 'X.npy',
        y_path: Union[str, Path] = 'y.npy',
        lang_path: Union[str, Path] = 'lang.json',
        test_save_path: Optional[Union[str, Path]] = None
) -> Optional[str]:
    X_train, X_dev, X_test, y_train, y_dev, y_test, lang_input, lang_output = read_data(x_path, y_path, lang_path)

    if test_save_path is not None:
        if isinstance(test_save_path, str):
            test_save_path = Path(test_save_path)
        elif not isinstance(test_save_path, Path):
            raise ValueError("test_save_path must be a string or a Path object")

        buf = test_save_path.open(mode="w", encoding="utf-8")
    else:
        buf = StringIO()

    for i in range(len(X_test)):
        buf.write(f"{lang_input.sentence_from_indices(X_test[i])}\t{lang_output.sentence_from_indices(y_test[i])}\n")

    if isinstance(buf, StringIO):
        return buf.getvalue()

    buf.close()
    return None

if __name__ == '__main__':
    pass
    # from model import paths
    #
    # pho = True
    # json_ = False
    # json_mode = 2
    # suffixes = ["_ly", "_midi", "_notes"]
    # max_lens = [800, 800, 150]
    #
    # midi_seps = ([",", " ", "[", "]"], "-")
    # ly_seps = ([" ", "\\", "\n", "{", "}", "|", "(", ")", "[", "]"], "-")
    # notes_seps = (" | ", "-")
    # seps = [ly_seps, midi_seps, notes_seps]
    #
    # l1_sep, l2_sep = seps[json_mode]
    # suffix = "" if not json_ else suffixes[json_mode]
    # max_len = 1000 if not json else max_lens[json_mode]
    #
    # params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path = paths(pho, suffix, json_)
    #
    # print(params_path, model_path, og_lang_path, x_data, y_data, lang_path, eval_path)
    #
    # # if json_:
    # #     X, y, l1, l2 = Language.read_data_from_json(og_lang_path, max_length=max_len , l1_sep=l1_sep, l2_sep=l2_sep, reverse=True)
    # # else:
    # #     X, y, l1, l2 = Language.read_data_from_txt(og_lang_path)
    # #
    # # Language.save_data(X, y, l1, l2, x_data, y_data, lang_path)
    #
    # extract_test_data(x_data, y_data, lang_path, test_save_path=Path("test.txt"))
    # print("Done")
    #
