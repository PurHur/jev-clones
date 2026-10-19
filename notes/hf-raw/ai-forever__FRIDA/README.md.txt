---
license: mit
language:
- ru
- en
tags:
- mteb
- transformers
- sentence-transformers
base_model: ai-forever/FRED-T5-1.7B
pipeline_tag: feature-extraction
datasets:
- ai-forever/solyanka
---

# Model Card for FRIDA

<figure>
  <img src="img.jpg">
</figure>

FRIDA is a full-scale finetuned general text embedding model inspired by denoising architecture based on T5. The model is based on the encoder part of [FRED-T5](https://arxiv.org/abs/2309.10931) model and continues research of text embedding models ([ruMTEB](https://arxiv.org/abs/2408.12503), [ru-en-RoSBERTa](https://huggingface.co/ai-forever/ru-en-RoSBERTa)). It has been pre-trained on a Russian-English dataset and fine-tuned for improved performance on the target task. 

For more model details please refer to our [article](https://habr.com/ru/companies/sberdevices/articles/909924/) (RU). The model's results are presented on the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) and [rusBEIR](https://huggingface.co/spaces/kaengreg/rusBEIR) leaderboards.

## Usage

The model can be used as is with prefixes. It is recommended to use CLS pooling. The choice of prefix and pooling depends on the task. 

We use the following basic rules to choose a prefix:
- `"search_query: "` and `"search_document: "` prefixes are for answer or relevant paragraph retrieval
- `"paraphrase: "` prefix is for symmetric paraphrasing related tasks (STS, paraphrase mining, deduplication)
- `"categorize: "` prefix is for asymmetric matching of document title and body (e.g. news, scientific papers, social posts)
- `"categorize_sentiment: "` prefix is for any tasks that rely on sentiment features (e.g. hate, toxic, emotion)
- `"categorize_topic: "` prefix is intended for tasks where you need to group texts by topic
- `"categorize_entailment: "` prefix is for textual entailment task (NLI)

To better tailor the model to your needs, you can fine-tune it with relevant high-quality Russian and English datasets.

Below are examples of texts encoding using the Transformers and SentenceTransformers libraries.
