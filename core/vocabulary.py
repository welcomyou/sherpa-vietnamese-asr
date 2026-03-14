import codecs
from collections import defaultdict
import logging
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union, TYPE_CHECKING
from filelock import FileLock


logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = "non_padded_namespaces.txt"
_NEW_LINE_REGEX = re.compile(r"\n|\r\n")


def namespace_match(pattern: str, namespace: str):
    """
    Matches a namespace pattern against a namespace string.  For example, `*tags` matches
    `passage_tags` and `question_tags` and `tokens` matches `tokens` but not
    `stemmed_tokens`.
    """
    if pattern[0] == "*" and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


class _NamespaceDependentDefaultDict(defaultdict):
    """
    This is a [defaultdict]
    (https://docs.python.org/2/library/collections.html#collections.defaultdict) where the
    default value is dependent on the key that is passed.
    We use "namespaces" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    namespaces (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the namespace (the key used in the `defaultdict`), and use different
    default values depending on whether the namespace passes the filter.
    To do filtering, we take a set of `non_padded_namespaces`.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with `*`.  In other words, if `*tags` is in `non_padded_namespaces` then
    `passage_tags`, `question_tags`, etc. (anything that ends with `tags`) will have the
    `non_padded` default value.
    # Parameters
    non_padded_namespaces : `Iterable[str]`
        A set / list / tuple of strings describing which namespaces are not padded.  If a namespace
        (key) is missing from this dictionary, we will use :func:`namespace_match` to see whether
        the namespace should be padded.  If the given namespace matches any of the strings in this
        list, we will use `non_padded_function` to initialize the value for that namespace, and
        we will use `padded_function` otherwise.
    padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that `should` be
        padded.
    non_padded_function : `Callable[[], Any]`
        A zero-argument function to call to initialize a value for a namespace that should `not` be
        padded.
    """

    def __init__(
        self,
        non_padded_namespaces: Iterable[str],
        padded_function: Callable[[], Any],
        non_padded_function: Callable[[], Any],
    ) -> None:
        self._non_padded_namespaces = set(non_padded_namespaces)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super().__init__()

    def add_non_padded_namespaces(self, non_padded_namespaces: Set[str]):
        # add non_padded_namespaces which weren't already present
        self._non_padded_namespaces.update(non_padded_namespaces)


class _TokenToIndexDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(non_padded_namespaces, lambda: {padding_token: 0, oov_token: 1}, lambda: {})


class _IndexToTokenDefaultDict(_NamespaceDependentDefaultDict):
    def __init__(self, non_padded_namespaces: Set[str], padding_token: str, oov_token: str) -> None:
        super().__init__(non_padded_namespaces, lambda: {0: padding_token, 1: oov_token}, lambda: {})


class Vocabulary:
    def __init__(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> None:
        self._padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        self._oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        self._non_padded_namespaces = set(non_padded_namespaces)

        self._token_to_index = _TokenToIndexDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )
        self._index_to_token = _IndexToTokenDefaultDict(
            self._non_padded_namespaces, self._padding_token, self._oov_token
        )

    @classmethod
    def from_files(
        cls,
        directory: Union[str, os.PathLike],
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> "Vocabulary":
        """
        Loads a `Vocabulary` that was serialized either using `save_to_files` or inside
        a model archive file.
        # Parameters
        directory : `str`
            The directory or archive file containing the serialized vocabulary.
        """
        logger.info("Loading token dictionary from %s.", directory)
        padding_token = padding_token if padding_token is not None else DEFAULT_PADDING_TOKEN
        oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN

        if not os.path.isdir(directory):
            raise ValueError(f"{directory} not exist")

        # We use a lock file to avoid race conditions where multiple processes
        # might be reading/writing from/to the same vocab files at once.
        with FileLock(os.path.join(directory, ".lock")):
            with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), "r", "utf-8") as namespace_file:
                non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

            vocab = cls(
                non_padded_namespaces=non_padded_namespaces,
                padding_token=padding_token,
                oov_token=oov_token,
            )

            # Check every file in the directory.
            for namespace_filename in os.listdir(directory):
                if namespace_filename == NAMESPACE_PADDING_FILE:
                    continue
                if namespace_filename.startswith("."):
                    continue
                namespace = namespace_filename.replace(".txt", "")
                if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                    is_padded = False
                else:
                    is_padded = True
                filename = os.path.join(directory, namespace_filename)
                if not os.path.isfile(filename):
                    continue
                vocab.set_from_file(filename, is_padded, namespace=namespace, oov_token=oov_token)

        return vocab

    @classmethod
    def empty(cls) -> "Vocabulary":
        """
        This method returns a bare vocabulary instantiated with `cls()` (so, `Vocabulary()` if you
        haven't made a subclass of this object).  The only reason to call `Vocabulary.empty()`
        instead of `Vocabulary()` is if you are instantiating this object from a config file.  We
        register this constructor with the key "empty", so if you know that you don't need to
        compute a vocabulary (either because you're loading a pre-trained model from an archive
        file, you're using a pre-trained transformer that has its own vocabulary, or something
        else), you can use this to avoid having the default vocabulary construction code iterate
        through the data.
        """
        return cls()

    def set_from_file(
        self,
        filename: str,
        is_padded: bool = True,
        oov_token: str = DEFAULT_OOV_TOKEN,
        namespace: str = "tokens",
    ):
        """
        If you already have a vocabulary file for a trained model somewhere, and you really want to
        use that vocabulary file instead of just setting the vocabulary from a dataset, for
        whatever reason, you can do that with this method.  You must specify the namespace to use,
        and we assume that you want to use padding and OOV tokens for this.
        # Parameters
        filename : `str`
            The file containing the vocabulary to load.  It should be formatted as one token per
            line, with nothing else in the line.  The index we assign to the token is the line
            number in the file (1-indexed if `is_padded`, 0-indexed otherwise).  Note that this
            file should contain the OOV token string!
        is_padded : `bool`, optional (default=`True`)
            Is this vocabulary padded?  For token / word / character vocabularies, this should be
            `True`; while for tag or label vocabularies, this should typically be `False`.  If
            `True`, we add a padding token with index 0, and we enforce that the `oov_token` is
            present in the file.
        oov_token : `str`, optional (default=`DEFAULT_OOV_TOKEN`)
            What token does this vocabulary use to represent out-of-vocabulary characters?  This
            must show up as a line in the vocabulary file.  When we find it, we replace
            `oov_token` with `self._oov_token`, because we only use one OOV token across
            namespaces.
        namespace : `str`, optional (default=`"tokens"`)
            What namespace should we overwrite with this vocab file?
        """
        if is_padded:
            self._token_to_index[namespace] = {self._padding_token: 0}
            self._index_to_token[namespace] = {0: self._padding_token}
        else:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}
        with codecs.open(filename, "r", "utf-8") as input_file:
            lines = _NEW_LINE_REGEX.split(input_file.read())
            # Be flexible about having final newline or not
            if lines and lines[-1] == "":
                lines = lines[:-1]
            for i, line in enumerate(lines):
                index = i + 1 if is_padded else i
                token = line.replace("@@NEWLINE@@", "\n")
                if token == oov_token:
                    token = self._oov_token
                self._token_to_index[namespace][token] = index
                self._index_to_token[namespace][index] = token
        if is_padded:
            assert self._oov_token in self._token_to_index[namespace], "OOV token not found!"

    def add_token_to_namespace(self, token: str, namespace: str = "tokens") -> int:
        """
        Adds `token` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings, or saving and loading will break."
                "  Got %s (with type %s)" % (repr(token), type(token))
            )
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            return index
        else:
            return self._token_to_index[namespace][token]

    def add_tokens_to_namespace(self, tokens: List[str], namespace: str = "tokens") -> List[int]:
        """
        Adds `tokens` to the index, if they are not already present.  Either way, we return the
        indices of the tokens in the order that they were given.
        """
        return [self.add_token_to_namespace(token, namespace) for token in tokens]

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        try:
            return self._token_to_index[namespace][token]
        except KeyError:
            try:
                return self._token_to_index[namespace][self._oov_token]
            except KeyError:
                logger.error("Namespace: %s", namespace)
                logger.error("Token: %s", token)
                raise KeyError(
                    f"'{token}' not found in vocab namespace '{namespace}', and namespace "
                    f"does not contain the default OOV token ('{self._oov_token}')"
                )

    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        return self._index_to_token[namespace][index]

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        return len(self._token_to_index[namespace])

    def get_namespaces(self) -> Set[str]:
        return set(self._index_to_token.keys())
