"""MATILDA multimodal aliases with an explicit loader for the frozen preprocessing configuration."""
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
from .tokenization_matilda_jev import MatildaJevTokenizer


class MatildaJevImageProcessor(Qwen2VLImageProcessor):
    pass


class MatildaJevVideoProcessor(Qwen3VLVideoProcessor):
    pass


class MatildaJevProcessor(Qwen3VLProcessor):
    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        if processor_dict is None:
            raise ValueError("MATILDA requires its processor_config.json")
        image = dict(processor_dict["image_processor"])
        video = dict(processor_dict["video_processor"])
        image.pop("image_processor_type", None)
        video.pop("video_processor_type", None)
        image.pop("auto_map", None)
        video.pop("auto_map", None)
        tokenizer = MatildaJevTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        components = {
            "image_processor": MatildaJevImageProcessor(**image),
            "tokenizer": tokenizer,
            "video_processor": MatildaJevVideoProcessor(**video),
        }
        return [components[name] for name in cls.get_attributes()]


MatildaJevProcessor.register_for_auto_class("AutoProcessor")
