# Context Attribution

## Model Summary

**Context Attribution** is a LoRA purpose-built for a base model to predict the sentences in the context that were most important for the base model to generate each sentence in its response or output. We provide LoRA adapters for the following base models: Granite 4.0 Micro, Granite 4.1 3B, Granite 4.1 8B, Granite 4.1 30B. Here, context includes previous conversation turns as well as any documents provided to the base model. The context attribution LoRA thus helps to explain the base model's behavior, specifically how its probability of generating a certain response sentence is affected by different parts of the context.

The context attribution model takes the form of a [LoRA adapter](https://arxiv.org/abs/2106.09685). The adapter greatly improves the base model's ability to attribute to context (see Evaluation section below) while retaining other capabilities of the base model. The context attribution adapter identifies which context sentences actually influenced the base model’s response, while IBM Granite’s [citation generation adapter](https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/main/citations/README.md) highlights sentences that support the response regardless of whether the model used them.

- **Developer:** IBM Research
- **HF Collection:** [Granite Libraries](https://huggingface.co/collections/ibm-granite/granite-libraries)
- **GitHub Repository:** https://github.com/ibm-granite
- **Release Date:** May 5th, 2026
- **Model Type:** LoRA adapters for:
  - [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro)
  - [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b)
  - [ibm-granite/granite-4.1-8b](https://huggingface.co/ibm-granite/granite-4.1-8b)
  - [ibm-granite/granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Paper:** Context Attribution is trained to approximate the importance ranking of context sentences provided by the MExGen method in [[Monteiro Paes and Wei et al., ACL 2025] Multi-Level Explanations for Generative Language Models](https://aclanthology.org/2025.acl-long.1553/).


## Usage

**Intended use:** Context Attribution is a context attribution adapter intended for a specific base model. It enables the base model to accurately identify which sentences in the context (including previous conversation turns and documents) were most important to the model when generating each sentence in its response. The adapter thus helps to explain the base model's behavior, specifically how its probability of generating a certain response sentence is affected by parts of the context. This adapter is designed to be used as part of the Granite inference pipeline. It is intended to be called after the base model generates a response to provide a post hoc explanation.

The context attribution adapter is similar to the [citations adapter](https://huggingface.co/ibm-granite/granitelib-rag-r1.0/blob/main/citations/README.md) found in the [granitelib-rag-r1.0](https://huggingface.co/ibm-granite/granitelib-rag-r1.0) library, in that it uses a comparable input/output format and similiar training data. However, the two differ in the type of attribution they provide. Citation generation provides corroborative attribution, identifying document sentences that best support a response regardless of whether the model relied on them, making it model-agnostic. In contrast, the context attribution adapter provides contributive attribution, identifying the context sentences that actually influenced the base model’s response. It also includes prior conversation turns (not just documents) and ranks context sentences by importance rather than listing them unordered.

**Input Format** The context attribution LoRA expects the input to be processed as follows: 1) The last assistant response (the response to be attributed) should be split into sentences and the sentences numbered with tags as follows: `"<r0> sentence 0 <r1> sentence 1 ... "`. 2) The context (documents and previous conversation turns) should also be split into sentences and tagged as `"<c0> sentence 0 <c1> sentence 1 ... "`. The numbering of context sentences starts with the first document (if present) and continues as a single sequence through all the documents and then previous conversation turns, in that order. This input processing is performed automatically if the context attribution adapter is called using IBM's [mellea](https://github.com/generative-computing/mellea) package, as the Quickstart Example below shows. Alternatively, the user can insert sentence tags manually or by other means (also shown in the Quickstart Example below).

**Output Format** The context attribution LoRA produces a JSON array of the form `[{"r": 0, "c": [3, 1, 4]}, {"r": 1, "c": [1, 5]}, ... ]`. Each JSON object in the array corresponds to a response sentence and has two members: `"r"` with the response sentence number as the value, and `"c"` with an array of context sentence numbers as the value. The context sentences are listed in order of decreasing importance for each response sentence (in the example above, context sentence 3 is identified as the most important for the base model to generate response sentence 0).

**Use Cases**
* Human understanding: Context attribution helps to explain a base LLM by showing which parts of the context were most important to it in generating a certain response. Such views into the LLM's behavior may help users calibrate their trust in using the LLM in a particular domain.
* RAG: The Context Attribution adapter is trained on a retrieval-augmented generation (RAG) dataset, specifically on document-grounded multi-turn question answering. The Evaluation section below shows that it also performs well when the use case is document summarization. It is thus especially suitable for RAG scenarios where users may wish to attribute LLM responses to a sizable context.
* Context reliance assessment: Context attribution can also be used to assess whether an LLM makes use of its context at all for a specific response. Low attribution scores to all context sentences might mean one of several things: 1) The response sentence is strongly implied by previous response sentences and does not need to refer to context; 2) the response comes primarily from the LLM's parametric knowledge; 3) the response is hallucinated. As such, determining the main reason for context non-reliance would require further investigation.


### Quickstart Example (LoRA)

We first define the inputs (documents and conversation turns) for the example.
```python
# Documents provided as context
documents = [
    {"doc_id": "0", "text": "The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 as a vehicle for their collaborative and solo projects. Lennon and Ono had begun a personal and artistic relationship in 1968, collaborating on several experimental releases. Following their marriage in 1969, they decided that all of their future endeavours would be credited to a conceptual and collaborative vehicle, Plastic Ono Band. The band would go on to feature a rotating lineup of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former Beatles bandmates George Harrison and Ringo Starr. Lennon and Ono left the UK to settle in New York City during the fall of 1971. In Greenwich Village, the couple became more politically active and began writing protest songs. These songs became the basis for their next album, Some Time in New York City. As backing, they enlisted the help of New York band Elephant's Memory, consisting of guitarist Wayne 'Tex' Gabriel, bassist Gary Van Scyoc, saxophonist Stan Bronstein, keyboardist Adam Ippolito, keyboardist John La Boosca, and drummer Richard Frank, Jr. Phil Spector produced, and Jim Keltner also played on the album. The album was released on 12 June 1972, credited to \"John & Yoko/Plastic Ono Band with Elephant's Memory plus Invisible Strings\". Some Time in New York City included a second disc, entitled Live Jam, which included the recordings from the 1969 Peace for Christmas concert and the 1971 performance with Frank Zappa. Ono and Lennon continued their work with Elephant's Memory throughout 1972, performing as the Plastic Ono Elephant's Memory Band (which also included Jim Keltner). On 30 August, they performed a pair of benefit concerts at Madison Square Garden. The benefit, entitled \"One to One\", was organised by Geraldo Rivera to raise money for children with mental challenges. By this time, La Boosca had departed the band, and the concert saw the addition of John Ward on bass. The concert was filmed and recorded, later released in February 1986 as the album Live In New York City. They also performed at the Jerry Lewis MDA Labor Day Telethon. The last collaboration of the Plastic Ono Elephant's Memory Band was Ono's double album Approximately Infinite Universe. It was recorded throughout the fall of 1972, and was released in January 1973."},
    {"doc_id": "1", "text": "The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 as a vehicle for their collaborative and solo projects. Lennon and Ono had begun a personal and artistic relationship in 1968, collaborating on several experimental releases. Following their marriage in 1969, they decided that all of their future endeavours would be credited to a conceptual and collaborative vehicle, Plastic Ono Band. The band would go on to feature a rotating lineup of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former Beatles bandmates George Harrison and Ringo Starr. By the beginning of 1973, recording had begun in earnest on Ono's next album, Feeling the Space, featuring a new group of studio musicians. The newest incarnation of the Plastic Ono Band featured guitarist David Spinozza, keyboardist Ken Ascher, bassist Gordon Edwards, percussionists Arthur Jenkins and David Friedman, saxophonist Michael Brecker, pedal steel guitarist Sneaky Pete Kleinow, as well as regular contributor Jim Keltner. The album would be released in November. Throughout 1973, Lennon and Ono's relationship became strained. By August, the two had begun a period of separation that Lennon called \"The Lost Weekend\". Lennon began the recording of his own album, Mind Games, using the same players as on Feeling the Space, dubbed \"The Plastic U.F.Ono Band\". Around the time of the album's release in November, Lennon moved to Los Angeles with new lover May Pang. In October, Lennon began the recording of an album of rock 'n' roll oldies (a contractual obligation due to a lawsuit). These featured many Plastic Ono Band regulars (including much of the \"U.F.Ono Band\", Klaus Voorman, and the return of Phil Spector to the production chair), but upon release in 1975 as Rock 'n' Roll, it was credited to Lennon alone. The sessions for Rock 'n' Roll were extremely troubled, and the sessions were abandoned until a later date. In July 1974, Lennon returned to New York to record Walls and Bridges. The new \"Plastic Ono Nuclear Band\" featured both old and new faces, with Jim Keltner, Kenneth Ascher, and Arthur Jenkins continuing from Mind Games, the returns of Klaus Voorman, Nicky Hopkins, and Bobby Keys, and the addition of guitarists Jesse Ed Davis and Eddie Mottau. Recording was finished in August, and the album was released 26 September and 4 October in the US and UK respectively. Walls and Bridges would prove to be the last release of new material by the Plastic Ono Band in the 1970s. Lennon subsequently returned to his marriage with Ono and retired from music following the birth of his son Sean. The compilation Shaved Fish was released in October 1975, Lennon's last release credited to the Plastic Ono Band. Upon his and Ono's return to music in 1980 for the album Double Fantasy, they played with an all-new group of studio musicians who were not billed as any variation of the Plastic Ono Band name. Lennon was shot and killed shortly after the release of the album."},
    {"doc_id": "2", "text": "John Winston Ono Lennon  (9 October 1940 - 8 December 1980) was an English singer, songwriter, and peace activist who co-founded the Beatles, the most commercially successful band in the history of popular music. He and fellow member Paul McCartney formed a much-celebrated songwriting partnership. Along with George Harrison and Ringo Starr, the group would ascend to world-wide fame during the 1960s. During his marriage to Cynthia, Lennon's first son Julian was born at the same time that his commitments with the Beatles were intensifying at the height of Beatlemania. Lennon was touring with the Beatles when Julian was born on 8 April 1963. Julian's birth, like his mother Cynthia's marriage to Lennon, was kept secret because Epstein was convinced that public knowledge of such things would threaten the Beatles' commercial success. Julian recalled that as a small child in Weybridge some four years later, \"I was trundled home from school and came walking up with one of my watercolour paintings. It was just a bunch of stars and this blonde girl I knew at school. And Dad said, 'What's this?' I said, 'It's Lucy in the sky with diamonds.'\" Lennon used it as the title of a Beatles song, and though it was later reported to have been derived from the initials LSD, Lennon insisted, \"It's not an acid song.\" McCartney corroborated Lennon's explanation that Julian innocently came up with the name. Lennon was distant from Julian, who felt closer to McCartney than to his father. During a car journey to visit Cynthia and Julian during Lennon's divorce, McCartney composed a song, \"Hey Jules\", to comfort him. It would evolve into the Beatles song \"Hey Jude\". Lennon later said, \"That's his best song. It started off as a song about my son Julian ... he turned it into 'Hey Jude'. I always thought it was about me and Yoko but he said it wasn't.\" Lennon's relationship with Julian was already strained, and after Lennon and Ono moved to Manhattan in 1971, Julian would not see his father again until 1973. With Pang's encouragement, arrangements were made for Julian (and his mother) to visit Lennon in Los Angeles, where they went to Disneyland. Julian started to see his father regularly, and Lennon gave him a drumming part on a Walls and Bridges track. He bought Julian a Gibson Les Paul guitar and other instruments, and encouraged his interest in music by demonstrating guitar chord techniques. Julian recalls that he and his father \"got on a great deal better\" during the time he spent in New York: \"We had a lot of fun, laughed a lot and had a great time in general.\" In a Playboy interview with David Sheff shortly before his death, Lennon said, \"Sean was a planned child, and therein lies the difference. I don't love Julian any less as a child. He's still my son, whether he came from a bottle of whiskey or because they didn't have pills in those days. He's here, he belongs to me, and he always will.\" He said he was trying to re-establish a connection with the then 17-year-old, and confidently predicted, \"Julian and I will have a relationship in the future.\" After his death it was revealed that he had left Julian very little in his will."},
]

# Previous conversation turns
messages = [
    {"role": "user", "content": "Who were the members of The Metal Ono Band, which was formed by Yoko Ono in 1976 to explore her interest in heavy metal music?"},
    {"role": "assistant", "content": "I'm sorry, but I don't have the data to answer that specific question. "},
]
# Current question
question = "What was the concept behind the formation of the Plastic Ono Band?"
messages.append({"role": "user", "content": question})

# Response pre-generated by granite-4.1-3b, could also generate a fresh response
response = "The Plastic Ono Band was formed by John Lennon and Yoko Ono in 1969 as a conceptual and collaborative vehicle for their collaborative and solo projects. Following their marriage in 1969, they decided that all of their future endeavors would be credited to this vehicle. The band was intended to serve as a platform for their artistic and personal expressions, featuring a rotating lineup of musicians over the years."
```

**Usage with Mellea**
```python
import json    # only for display

from mellea import model_ids, start_backend
from mellea.stdlib.components import Document, Message    # constructing the conversation
from mellea.stdlib.components.intrinsic.core import find_context_attributions    # intrinsic function for calling context attribution

ctx, backend = start_backend(
    "hf", model_id=model_ids.IBM_GRANITE_4_1_3B, context_type="chat"
)

# Previous conversation turns + current question
for message in messages:
    ctx = ctx.add(Message(message["role"], message["content"]))
# Documents
documents_mellea = [Document(**document) for document in documents]

# Call context attribution intrinsic function
result = find_context_attributions(response, documents_mellea, ctx, backend)
print(f"Result of context attribution intrinsic:\n{json.dumps(result, indent=2)}")

# Prints an array of JSON objects where each object shows a response sentence,
# a context sentence to which the response sentence is attributed (in order of decreasing importance),
# and other fields such as the document ID where the context sentence is located.
# (You may get slightly different orderings of context sentences, depending on your hardware or environment.)
```

**Usage without Mellea**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_NAME = "ibm-granite/granite-4.1-3b"
ADAPTER_REPO = "ibm-granite/granitelib-core-r1.0"
ADAPTER_SUBFOLDER = "context-attribution/granite-4.1-3b/lora"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load context attribution model (base model + LoRA)
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME, padding_side="left", trust_remote_code=True)
model_ca = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE_NAME, device_map="auto", dtype=torch.bfloat16),
    ADAPTER_REPO,
    subfolder=ADAPTER_SUBFOLDER
)

# The context attribution adapter expects the last assistant response and context (documents
# + previous conversation turns) to be split into sentences and the sentences to be tagged with numbers.
# In this snippet, we assume that this input processing has been done as below.
documents_rewritten = [
    {"doc_id": "0", "text": "<c0> The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 as a vehicle for their collaborative and solo projects. <c1> Lennon and Ono had begun a personal and artistic relationship in 1968, collaborating on several experimental releases. <c2> Following their marriage in 1969, they decided that all of their future endeavours would be credited to a conceptual and collaborative vehicle, Plastic Ono Band. <c3> The band would go on to feature a rotating lineup of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former Beatles bandmates George Harrison and Ringo Starr. <c4> Lennon and Ono left the UK to settle in New York City during the fall of 1971. <c5> In Greenwich Village, the couple became more politically active and began writing protest songs. <c6> These songs became the basis for their next album, Some Time in New York City. <c7> As backing, they enlisted the help of New York band Elephant's Memory, consisting of guitarist Wayne 'Tex' Gabriel, bassist Gary Van Scyoc, saxophonist Stan Bronstein, keyboardist Adam Ippolito, keyboardist John La Boosca, and drummer Richard Frank, Jr. <c8> Phil Spector produced, and Jim Keltner also played on the album. <c9> The album was released on 12 June 1972, credited to \"John & Yoko/Plastic Ono Band with Elephant's Memory plus Invisible Strings\". <c10> Some Time in New York City included a second disc, entitled Live Jam, which included the recordings from the 1969 Peace for Christmas concert and the 1971 performance with Frank Zappa. <c11> Ono and Lennon continued their work with Elephant's Memory throughout 1972, performing as the Plastic Ono Elephant's Memory Band (which also included Jim Keltner). <c12> On 30 August, they performed a pair of benefit concerts at Madison Square Garden. <c13> The benefit, entitled \"One to One\", was organised by Geraldo Rivera to raise money for children with mental challenges. <c14> By this time, La Boosca had departed the band, and the concert saw the addition of John Ward on bass. <c15> The concert was filmed and recorded, later released in February 1986 as the album Live In New York City. <c16> They also performed at the Jerry Lewis MDA Labor Day Telethon. <c17> The last collaboration of the Plastic Ono Elephant's Memory Band was Ono's double album Approximately Infinite Universe. <c18> It was recorded throughout the fall of 1972, and was released in January 1973."},
    {"doc_id": "1", "text": "<c19> The Plastic Ono Band is a band formed by John Lennon and Yoko Ono in 1969 as a vehicle for their collaborative and solo projects. <c20> Lennon and Ono had begun a personal and artistic relationship in 1968, collaborating on several experimental releases. <c21> Following their marriage in 1969, they decided that all of their future endeavours would be credited to a conceptual and collaborative vehicle, Plastic Ono Band. <c22> The band would go on to feature a rotating lineup of many musicians, including Eric Clapton, Klaus Voormann, Alan White, Billy Preston, Jim Keltner, Delaney & Bonnie and Friends, and Lennon's former Beatles bandmates George Harrison and Ringo Starr. <c23> By the beginning of 1973, recording had begun in earnest on Ono's next album, Feeling the Space, featuring a new group of studio musicians. <c24> The newest incarnation of the Plastic Ono Band featured guitarist David Spinozza, keyboardist Ken Ascher, bassist Gordon Edwards, percussionists Arthur Jenkins and David Friedman, saxophonist Michael Brecker, pedal steel guitarist Sneaky Pete Kleinow, as well as regular contributor Jim Keltner. <c25> The album would be released in November. <c26> Throughout 1973, Lennon and Ono's relationship became strained. <c27> By August, the two had begun a period of separation that Lennon called \"The Lost Weekend\". <c28> Lennon began the recording of his own album, Mind Games, using the same players as on Feeling the Space, dubbed \"The Plastic U.F.Ono Band\". <c29> Around the time of the album's release in November, Lennon moved to Los Angeles with new lover May Pang. <c30> In October, Lennon began the recording of an album of rock 'n' roll oldies (a contractual obligation due to a lawsuit). <c31> These featured many Plastic Ono Band regulars (including much of the \"U.F.Ono Band\", Klaus Voorman, and the return of Phil Spector to the production chair), but upon release in 1975 as Rock 'n' Roll, it was credited to Lennon alone. <c32> The sessions for Rock 'n' Roll were extremely troubled, and the sessions were abandoned until a later date. <c33> In July 1974, Lennon returned to New York to record Walls and Bridges. <c34> The new \"Plastic Ono Nuclear Band\" featured both old and new faces, with Jim Keltner, Kenneth Ascher, and Arthur Jenkins continuing from Mind Games, the returns of Klaus Voorman, Nicky Hopkins, and Bobby Keys, and the addition of guitarists Jesse Ed Davis and Eddie Mottau. <c35> Recording was finished in August, and the album was released 26 September and 4 October in the US and UK respectively. <c36> Walls and Bridges would prove to be the last release of new material by the Plastic Ono Band in the 1970s. <c37> Lennon subsequently returned to his marriage with Ono and retired from music following the birth of his son Sean. <c38> The compilation Shaved Fish was released in October 1975, Lennon's last release credited to the Plastic Ono Band. <c39> Upon his and Ono's return to music in 1980 for the album Double Fantasy, they played with an all-new group of studio musicians who were not billed as any variation of the Plastic Ono Band name. <c40> Lennon was shot and killed shortly after the release of the album."},
    {"doc_id": "2", "text": "<c41> John Winston Ono Lennon  (9 October 1940 - 8 December 1980) was an English singer, songwriter, and peace activist who co-founded the Beatles, the most commercially successful band in the history of popular music. <c42> He and fellow member Paul McCartney formed a much-celebrated songwriting partnership. <c43> Along with George Harrison and Ringo Starr, the group would ascend to world-wide fame during the 1960s. <c44> During his marriage to Cynthia, Lennon's first son Julian was born at the same time that his commitments with the Beatles were intensifying at the height of Beatlemania. <c45> Lennon was touring with the Beatles when Julian was born on 8 April 1963. <c46> Julian's birth, like his mother Cynthia's marriage to Lennon, was kept secret because Epstein was convinced that public knowledge of such things would threaten the Beatles' commercial success. <c47> Julian recalled that as a small child in Weybridge some four years later, \"I was trundled home from school and came walking up with one of my watercolour paintings. <c48> It was just a bunch of stars and this blonde girl I knew at school. <c49> And Dad said, 'What's this?' <c50> I said, 'It's Lucy in the sky with diamonds.'\" <c51> Lennon used it as the title of a Beatles song, and though it was later reported to have been derived from the initials LSD, Lennon insisted, \"It's not an acid song.\" <c52> McCartney corroborated Lennon's explanation that Julian innocently came up with the name. <c53> Lennon was distant from Julian, who felt closer to McCartney than to his father. <c54> During a car journey to visit Cynthia and Julian during Lennon's divorce, McCartney composed a song, \"Hey Jules\", to comfort him. <c55> It would evolve into the Beatles song \"Hey Jude\". <c56> Lennon later said, \"That's his best song. <c57> It started off as a song about my son Julian ... he turned it into 'Hey Jude'. <c58> I always thought it was about me and Yoko but he said it wasn't.\" <c59> Lennon's relationship with Julian was already strained, and after Lennon and Ono moved to Manhattan in 1971, Julian would not see his father again until 1973. <c60> With Pang's encouragement, arrangements were made for Julian (and his mother) to visit Lennon in Los Angeles, where they went to Disneyland. <c61> Julian started to see his father regularly, and Lennon gave him a drumming part on a Walls and Bridges track. <c62> He bought Julian a Gibson Les Paul guitar and other instruments, and encouraged his interest in music by demonstrating guitar chord techniques. <c63> Julian recalls that he and his father \"got on a great deal better\" during the time he spent in New York: \"We had a lot of fun, laughed a lot and had a great time in general.\" <c64> In a Playboy interview with David Sheff shortly before his death, Lennon said, \"Sean was a planned child, and therein lies the difference. <c65> I don't love Julian any less as a child. <c66> He's still my son, whether he came from a bottle of whiskey or because they didn't have pills in those days. <c67> He's here, he belongs to me, and he always will.\" <c68> He said he was trying to re-establish a connection with the then 17-year-old, and confidently predicted, \"Julian and I will have a relationship in the future.\" <c69> After his death it was revealed that he had left Julian very little in his will."},
]
messages_rewritten = [
    {"role": "user", "content": "<c70> Who were the members of The Metal Ono Band, which was formed by Yoko Ono in 1976 to explore her interest in heavy metal music?"},
    {"role": "assistant", "content": "<c71> I'm sorry, but I don't have the data to answer that specific question."},
    {"role": "user", "content": "<c72> What was the concept behind the formation of the Plastic Ono Band?"},
    {"role": "assistant", "content": "<r0> The Plastic Ono Band was formed by John Lennon and Yoko Ono in 1969 as a conceptual and collaborative vehicle for their collaborative and solo projects. <r1> Following their marriage in 1969, they decided that all of their future endeavors would be credited to this vehicle. <r2> The band was intended to serve as a platform for their artistic and personal expressions, featuring a rotating lineup of musicians over the years."},
    {"role": "user", "content": "You provided the last assistant response above based on context, which may include documents and/or previous conversation turns. Your response is divided into sentences, numbered in the format <r0> sentence 0 <r1> sentence 1 ... Sentences in the context are also numbered: <c0> sentence 0 <c1> sentence 1 ... For each response sentence, please list the context sentences that were most important for you to generate the response sentence. Provide your answer in JSON format, as an array of JSON objects, where each object has two members: \"r\" with the response sentence number as the value, and \"c\" with an array of context sentence numbers as the value. An example of such an array of objects is [{\"r\": 0, \"c\": [3, 1, 4]}, {\"r\": 1, \"c\": [1, 5]}]. List the context sentences in order from most important to least important. Ensure that you include an object for each response sentence, even if the corresponding array of context sentence numbers is empty. Answer with only the JSON and do not explain.\n"},
]

# Call context attribution model to attribute each response sentence to context sentences
inputs = tokenizer.apply_chat_template(messages_rewritten, documents=documents_rewritten, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(device)
output_tokens = model_ca.generate(**inputs, max_new_tokens=4096, do_sample=False)
output_text = tokenizer.decode(output_tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(output_text)

# [{"r": 0, "c": [0, 2, 71, 3, 7, 24, 34, 31, 22, 41]}, {"r": 1, "c": [2, 0, 3, 7, 24, 34, 31, 22, 41, 63]}, {"r": 2, "c": [3, 2, 7, 24, 34, 31, 22, 41, 63, 39]}]
# This raw output from the adapter means that response sentences 0, 1, and 2 are each attributed to many context sentences in decreasing order of importance.
# The most important context sentences for response sentences 0, 1, 2 are 0, 2, 3 respectively.
# (You may get slightly different orderings of context sentences, depending on your hardware or environment.)
```

## Evaluation

The context attribution adapters were evaluated on four datasets: MD2D-QUAC, ELI5, CNN/Daily Mail (CNN/DM), and XSum. The first two are document-grounded, multi-turn question answering datasets and the last two are summarization datasets. MD2D-QUAC evaluates the adpaters on a held-out test split of the dataset used for training (please see "Training Data" below for more details), ELI5 evaluates generalization to a different dataset within the same task, and CNN/DM and XSum evaluate generalization to a different task.

The evaluation metric is the area under the perturbation curve (AUPC), a standard metric for evaluating the faithfulness of a feature-attribution-like explanation to the model being explained. Please see the [MExGen paper](https://aclanthology.org/2025.acl-long.1553/) for more details on AUPC. Here, the perturbation curves are first multiplied by a linearly decreasing weight function, which decreases from 1 at 0% perturbation to 0 at 20% perturbation, yielding the weighted area under the perturbation curve (WAUPC).

Below we show tables of WAUPC values for each base model for which a context attribution LoRA is available. Since the context attribution task is model-dependent (i.e., to explain the base model's behavior), values may not be comparable across tables. Values are normalized and can be compared within a table as described below.

#### Granite 4.0 Micro

| method                     |   MD2D-QUAC |   ELI5 |   CNN/DM |   XSum |   average |
|:---------------------------|------------:|-------:|---------:|-------:|----------:|
| LOO *(skyline)*                        |      1.0000 | 1.0000 |   1.0000 | 1.0000 |    1.0000 |
| LOO thresholded *(realistic skyline)*           |      0.9780 | 0.9812 |   0.9976 | 0.9983 |    0.9888 |
| prompt granite-4.0-micro 0-shot              |      0.1476 | 0.1657 |   0.1629 | 0.2369 |    0.1783 |
| prompt granite-4.0-micro 1-shot              |      0.1982 | 0.2171 |   0.1693 | 0.2014 |    0.1965 |
| prompt GPT-OSS-120B 0-shot |      0.8619 | 0.8063 |   0.8735 | 0.6679 |    0.8024 |
| prompt GPT-OSS-120B 1-shot |      0.8779 | 0.8498 |   0.8996 | 0.8642 |    0.8729 |
| **context attribution LoRA**          |      **0.9360** | **0.9045** |   **0.9158** | **0.9223** |    **0.9197** |

#### Granite 4.1 3B

| method | MD2D-QUAC | ELI5 | CNN/DM | XSum | average |
|:---|---:|---:|---:|---:|---:|
| LOO *(skyline)* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LOO thresholded *(realistic skyline)* | 0.9908 | 0.9988 | 1.0004 | 1.0007 | 0.9977 |
| prompt granite-4.1-3b 0-shot | 0.4058 | 0.3918 | 0.2773 | 0.2808 | 0.3389 |
| prompt granite-4.1-3b 2-shot | 0.5032 | 0.4931 | 0.3395 | 0.3597 | 0.4239 |
| prompt GPT-OSS-120B 0-shot | 0.9040 | 0.8615 | 0.8911 | 0.7155 | 0.8430 |
| prompt GPT-OSS-120B 2-shot | 0.8909 | 0.8754 | 0.9050 | 0.8775 | 0.8872 |
| **context attribution LoRA** | **0.9053** | **0.8996** | **0.9310** | **0.9421** | **0.9195** |

#### Granite 4.1 8B

| method | MD2D-QUAC | ELI5 | CNN/DM | XSum | average |
|:---|---:|---:|---:|---:|---:|
| LOO *(skyline)* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LOO thresholded *(realistic skyline)* | 0.9854 | 0.9978 | 1.0000 | 0.9999 | 0.9958 |
| prompt granite-4.1-8b 0-shot | 0.3938 | 0.2528 | 0.2109 | 0.2179 | 0.2688 |
| prompt granite-4.1-8b 2-shot | 0.5186 | 0.4071 | 0.3238 | 0.2944 | 0.3860 |
| prompt GPT-OSS-120B 0-shot | 0.8681 | 0.8005 | 0.8762 | 0.7009 | 0.8114 |
| prompt GPT-OSS-120B 2-shot | 0.8648 | 0.8215 | 0.8938 | 0.8720 | 0.8631 |
| **context attribution LoRA** | **0.9408** | **0.9278** | **0.9756** | **0.9808** | **0.9563** |

#### Granite 4.1 30B

| method | MD2D-QUAC | ELI5 | CNN/DM | XSum | average |
|:---|---:|---:|---:|---:|---:|
| LOO *(skyline)* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LOO thresholded *(realistic skyline)* | 0.9877 | 0.9994 | 1.0003 | 1.0001 | 0.9969 |
| prompt granite-4.1-30b 0-shot | 0.6953 | 0.5896 | 0.6709 | 0.6452 | 0.6502 |
| prompt granite-4.1-30b 3-shot | 0.8089 | 0.6312 | 0.7712 | 0.7607 | 0.7430 |
| prompt GPT-OSS-120B 0-shot | 0.9098 | 0.7956 | 0.8706 | 0.6993 | 0.8188 |
| prompt GPT-OSS-120B 1-shot | 0.8970 | 0.7925 | 0.8789 | 0.8478 | 0.8540 |
| **context attribution LoRA** | **0.9188** | **0.8647** | **0.9123** | **0.9163** | **0.9030** |

The first row of each table shows the WAUPC for the leave-one-out (LOO) variant of the [MExGen method](https://aclanthology.org/2025.acl-long.1553/), which the adapters aim to approximate. The second row corresponds to a thresholded version of LOO, which is the one that actually generates training data for the adapters. LOO and thresholded LOO thus represent "skyline" and "realistic skyline" methods that the adapters ideally would match. The WAUPC values in each table have been normalized by dividing by the WAUPC of LOO (for the particular base model). As seen in the last row of each table, the adapters do well in approximating LOO, attaining at least 90% of its WAUPC in all but one case and 90.3%-95.6% averaged across datasets.

The tables compare the context attribution adapters to baselines that prompt an LLM to perform the context attribution task. These include prompting the base model, with and without providing examples (0-shot/1-shot or more), and prompting GPT-OSS-120B, also with 0-shot or more. 1-shot, 2-shot, and 3-shot prompting were evaluated and the best-performing number of shots is shown for each prompted model. It is clear that the context attribution adapter improves the base model's ability to attribute its own responses to context. The adapter also outperforms GPT-OSS-120B, an LLM with around 4-40X the number of parameters (the LoRA adapter contributes a negligible number of parameters to the base model).

## Training Details

Context Attribution is a LoRA adapter trained to approximate the importance ranking of context sentences provided by the MExGen method in [[Monteiro Paes and Wei et al., ACL 2025] Multi-Level Explanations for Generative Language Models](https://aclanthology.org/2025.acl-long.1553/).

**Training Data:** The adapters were trained on a mixture of two datasets, [MultiDoc2Dial](https://huggingface.co/datasets/IBM/multidoc2dial) and [QUAC](https://huggingface.co/datasets/allenai/quac), which are both for document-grounded, multi-turn question answering. This mixture of datasets, referred to as **MD2D-QUAC**, was also used to train IBM Granite's [citation generation adapter](https://huggingface.co/ibm-granite/granite-lib-rag-r1.0/blob/main/citations/README.md). 4101 question-answer conversation rounds were used for training and 1024 for validation. These numbers of instances were deliberately chosen to be modest to demonstrate that a well-performing adapter can be trained using this limited amount of data.

The MultiDoc2Dial and QUAC datasets consist of sets of grounding documents and multi-turn question-answering conversations based on the set of documents. For training the context attribution adapters, each conversation round (consisting of one user question followed by an assistant response), together with any conversation rounds that precede it, was treated as a separate "instance". The context for each instance includes the documents for the conversation as well as all conversation rounds except the last round, which is treated as the current round for the instance. The base model was called to re-generate a response to the question in the current round (since the purpose of the adapter is to attribute the base model's responses). The leave-one-out (LOO) variant of [MExGen](https://aclanthology.org/2025.acl-long.1553/) was then used to attribute each response sentence to context sentences, yielding "gold" attribution scores. LOO is thus treated as the "skyline" method in the Evaluation section. The LOO attribution scores were thresholded to produce a controlled-length list of context sentences, specifically a list of context sentence numbers in decreasing order of importance. The context attribution adapters were trained to reproduce these lists for all response sentences, given the context, response, and an instruction as input.

### Adapter Configurations

| Property | Granite 4.0 Micro | Granite 4.1 3B | Granite 4.1 8B | Granite 4.1 30B |
|-----------|------|------|------|------|
| Base model | ibm-granite/granite-4.0-micro | ibm-granite/granite-4.1-3b | ibm-granite/granite-4.1-8b | ibm-granite/granite-4.1-30b |
| PEFT type | LoRA | LoRA | LoRA | LoRA |
| Rank (r) | 16 | 16 | 16 | 32 |
| Alpha | 32 | 32 | 32 | 64 |
| Target modules | q_proj, k_proj, v_proj | q_proj, k_proj, v_proj | q_proj, k_proj, v_proj | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Max completion tokens | 4096 | 4096 | 4096 | 4096 |
| KV cache | Supported | Supported | Supported | Supported |
<!-- | Output format | e.g. `[{"r": 0, "c": [3, 1, 4]}, ... ]` where `"c"` lists most important context sentences for response sentence `"r"`| -->

**Infrastructure:** The Context Attribution LoRA adapters were trained on a single NVIDIA H100-80GB GPU.

**Ethical Considerations:** Each context attribution adapter is designed specifically for its respective base model and was trained on its behavior. While it may be applied to other LLMs, it has not been validated for them. In addition, the context attributions may not always align with human judgments of which context sentences should matter.

## Citation
If you find these adapters useful, please cite the following work:
```
@inproceedings{monteiro-paes-wei-2025-mexgen,
    title = "Multi-Level Explanations for Generative Language Models",
    author = "Monteiro Paes, Lucas  and
      Wei, Dennis  and
      Do, Hyo Jin  and
      Strobelt, Hendrik  and
      Luss, Ronny  and
      Dhurandhar, Amit  and
      Nagireddy, Manish  and
      Natesan Ramamurthy, Karthikeyan  and
      Sattigeri, Prasanna  and
      Geyer, Werner  and
      Ghosh, Soumya",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    url = "https://aclanthology.org/2025.acl-long.1553/",
    pages = "32291--32317",
}
```

## Resources

- ⭐️ Learn about the latest updates with Granite: https://www.ibm.com/granite
- 📄 Get started with tutorials, best practices, and prompt engineering advice: https://www.ibm.com/granite/docs/
- 💡 Learn about the latest Granite learning resources: https://research.ibm.com/blog/granite-4-1-ai-foundation-models