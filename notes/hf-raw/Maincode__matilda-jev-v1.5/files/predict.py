"""Native JEV JSON-lines inference; accepts state/question or state/questions."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
from maincode_jev_serve.model import DecisionModel, answer
from maincode_jev_serve.decide import decide

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(ROOT))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-tokens',type=int,default=131072)
    p.add_argument('--token-budget',type=int,default=131072)
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    model=DecisionModel(checkpoint=args.checkpoint,device=args.device)
    for line in sys.stdin:
        if not line.strip():continue
        row=json.loads(line)
        questions=row.get('questions')
        if questions is None:questions={'answer':row['question']}
        probs,tokens=decide(model,row['state'],questions,temperature=model.temperature,
            max_tokens=args.max_tokens,token_budget=args.token_budget,batch_size=args.batch_size,
            images=row.get('images',()))
        print(json.dumps({'model':'MATILDA JEV by Maincode','answers':{k:answer(questions[k],v) for k,v in probs.items()},
                          'usage':{'input_tokens':tokens}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
