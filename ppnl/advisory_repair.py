"""Same frozen action model; optional prompt rules, never rerank or rewrite actions."""
from ppnl.core import ACTIONS
from ppnl.policy import NAMES,state_prompt
from part2.run_rollback import reported_prefix


class AdvisoryActionPolicy:
    def __init__(self,base):
        self.runner=base.runner;self.ids=base.ids;self.cache={}
        self.usage={'forward_batches':0,'scored_states':0}

    def scores(self,observations):
        import torch
        keys=[];missing={}
        for g,pos,history,feedback in observations:
            key=(g.layout_id,g.goal,pos,tuple(history[-8:]),feedback)
            keys.append(key)
            if key not in self.cache:
                text=state_prompt(g,pos,self.runner.config.get('policy_local_status',False))
                text+='\nContinuation rules: Choose the next action yourself. Prefer progress toward the goal. Avoid repeating a recent position or immediately reversing when another move can help; backtrack if necessary.\nRecent positions: '+str(history[-8:])+'.\nUnmodified verifier feedback: '+feedback
                missing.setdefault(key,text)
        items=list(missing.items());runner=self.runner;bs=runner.config['training']['eval_batch_size']
        for offset in range(0,len(items),bs):
            batch=items[offset:offset+bs]
            inputs=runner.tokenizer([v for _,v in batch],padding=True,pad_to_multiple_of=8,truncation=False,return_tensors='pt')
            if inputs.input_ids.shape[1]>runner.input_limit:raise ValueError('R rule prompt exceeds input limit; no truncation')
            decoder=torch.full((len(batch),1),runner.model.config.decoder_start_token_id,dtype=torch.long,device=runner.device)
            with torch.inference_mode(),runner.autocast():
                logits=runner.model(**inputs.to(runner.device),decoder_input_ids=decoder,use_cache=False).logits[:,0,self.ids]
            for (key,_),values in zip(batch,logits.float().cpu().tolist()):self.cache[key]=values
            self.usage['forward_batches']+=1;self.usage['scored_states']+=len(batch)
        return [self.cache[k] for k in keys]


def repair_batch(grids,candidates,feedbacks,base,advisory,enhanced=False,max_new=36,*,backoff=None,prompt_rules=None):
    """R observes transitions, stops on goal/illegal action; no correction/search.

    Prefix safety can reject a report but never supplies a corrected V diagnosis.
    Every generated action is the raw model argmax over the same four tokens.
    """
    # Explicit options separate prompt guidance from rollback; old callers retain their behavior.
    k=(36 if enhanced else 0) if backoff is None else backoff
    guided=enhanced if prompt_rules is None else prompt_rules
    if not isinstance(k,int) or k<0:raise ValueError('backoff must be a nonnegative integer')
    plans=list(candidates);statuses=[];paths={};histories={}
    for i,(g,c,fb) in enumerate(zip(grids,candidates,feedbacks)):
        prefix,status=reported_prefix(g,c,fb,k)
        statuses.append(status)
        if prefix is None:continue
        path=prefix.split();positions=[g.start]
        for action in path:
            dr,dc=ACTIONS[action];positions.append((positions[-1][0]+dr,positions[-1][1]+dc))
        plans[i]=prefix
        if positions[-1]==g.goal:statuses[i]='prefix_reaches_goal';continue
        paths[i]=path;histories[i]=positions
    for depth in range(max_new):
        active=list(paths)
        if not active:break
        scores=advisory.scores([(grids[i],histories[i][-1],histories[i],feedbacks[i]) for i in active]) if guided else base.scores([(grids[i],histories[i][-1]) for i in active])
        for i,values in zip(active,scores):
            action=max(NAMES,key=lambda a:values[NAMES.index(a)])
            paths[i].append(action);plans[i]=' '.join(paths[i])
            dr,dc=ACTIONS[action];pos=histories[i][-1];nxt=(pos[0]+dr,pos[1]+dc)
            if not grids[i].inside(nxt) or nxt in grids[i].obstacles:
                statuses[i]='model_illegal_action';del paths[i];continue
            histories[i].append(nxt)
            if nxt==grids[i].goal:statuses[i]='model_reaches_goal';del paths[i]
            else:statuses[i]='action_budget'
    return plans,statuses
