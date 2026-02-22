from transformers import PretrainedConfig


class Seq2LabelsConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Seq2LabelsModel`]. It is used to
    instantiate a Seq2Labels model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Seq2Labels architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`] or [`TFBertModel`].
        pretrained_name_or_path (`str`, *optional*, defaults to `bert-base-cased`):
            Pretrained BERT-like model path
        load_pretrained (`bool`, *optional*, defaults to `False`):
            Whether to load pretrained model from `pretrained_name_or_path`
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        predictor_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        special_tokens_fix (`bool`, *optional*, defaults to `False`):
            Whether to add additional tokens to the BERT's embedding layer.
    Examples:
    ```python
    >>> from transformers import BertModel, BertConfig
    >>> # Initializing a Seq2Labels style configuration
    >>> configuration = Seq2LabelsConfig()
    >>> # Initializing a model from the bert-base-uncased style configuration
    >>> model = Seq2LabelsModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bert"

    def __init__(
        self,
        pretrained_name_or_path="bert-base-cased",
        vocab_size=15,
        num_detect_classes=4,
        load_pretrained=False,
        initializer_range=0.02,
        pad_token_id=0,
        use_cache=True,
        predictor_dropout=0.0,
        special_tokens_fix=False,
        label_smoothing=0.0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.num_detect_classes = num_detect_classes
        self.pretrained_name_or_path = pretrained_name_or_path
        self.load_pretrained = load_pretrained
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.predictor_dropout = predictor_dropout
        self.special_tokens_fix = special_tokens_fix
        self.label_smoothing = label_smoothing
