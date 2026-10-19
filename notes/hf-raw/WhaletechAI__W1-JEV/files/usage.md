# W1-JEV usage

Requires Linux, Python 3.12, and a CUDA GPU with BF16 support and a driver
compatible with CUDA 12.8. Runtime versions: PyTorch 2.10.0, Transformers 4.57.6,
and Tokenizers 0.22.2. The included FP32 checkpoint is approximately 14.96 GB;
the runtime uses BF16 for inference.

## Download and start

```bash
mkdir -p W1-JEV
cd W1-JEV
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub==0.36.2
```

If the repository is private, run `hf auth login` with an account that has
access before downloading. Public, ungated downloads do not require login.

Download the complete repository, including the checkpoint, then start the
local service:

```bash
hf download WhaletechAI/W1-JEV --local-dir .
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python structured_server.py --checkpoint w1-jev.pt
```

Wait until the server prints `w1-jev ready at http://127.0.0.1:8011`.
Keep it running, then open a second terminal in the downloaded `W1-JEV`
directory and run:

```bash
source .venv/bin/activate
curl --fail http://127.0.0.1:8011/health
python examples/ask.py
```

The health check returns `{"status": "ready", "model": "w1-jev"}`. The example
client prints a JSON result containing `answers.answer`.

## Ask a question

Replace the question and options below. The server applies the [djev template](https://github.com/mmastrac/djev), labels the options `A, B, C, D, ...` in order, and reads the answer in `answer: label` format.

```python
import json
from urllib.request import Request, urlopen

state = {"question": "What is the capital of France?"}
options = ["Paris", "London", "Berlin", "Rome"]

schema = {"questions": [{
    "id": "answer",
    "type": "choice",
    "instructions": "Choose the correct answer to the question.",
    "options": options
}]}

body = {
    "model": "w1-jev",
    "messages": [
        {"role": "system", "content": json.dumps(schema)},
        {"role": "user", "content": json.dumps(state)}
    ]
}
request = Request(
    "http://127.0.0.1:8011/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}
)
with urlopen(request, timeout=600) as response:
    reply = json.load(response)

answers = json.loads(reply["choices"][0]["message"]["content"])["answers"]
print(answers["answer"]["label"])  # A, B, C, or D
```

For questions that need context, add it to `state`, for example `{"question": "...", "context": "..."}`. Use 2–26 options; the complete input must fit `model.max_seq_len` in `config.json` (4,096 tokens by default).
