import json,sys
from tokenizers import Tokenizer
S=json.load(open(sys.argv[1])); root=sys.argv[2]
toks={"Qwen3.5 (JevAlt, Turnike, OpenThai base, johannhartmann)":"mertkayacs__Wahler-4B","OpenThai (Qwen3.5 + added tokens, Thai CPT)":"iapp__OpenThai-SystemOne","Bielik v3 (basal 1.x)":"Remek__basal-1.5-mini","Gemma 4 (basal-omni)":"Remek__basal-omni-1.0-mini","FRIDA / FRED-T5":"ai-forever__FRIDA-Decisions","ModernBERT-ja (sokudan)":"GeneLab__sokudan-ja-310m"}
langs=list(S)
print("tokenizer\tvocab\t"+"\t".join(langs))
for name,d in toks.items():
    t=Tokenizer.from_file(f"{root}/{d}/tokenizer.json.txt")
    row=[str(len(t.encode(S[l],add_special_tokens=False).ids)) for l in langs]
    print(name+"\t"+str(t.get_vocab_size())+"\t"+"\t".join(row))
