from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib, re


@dataclass(slots=True)
class InformationNeed:
    """Retrieval-only investigation intent.

    V1.4.5 deliberately has no semantic ``satisfied`` state. Evidence acquisition is
    tracked here; semantic convergence belongs exclusively to EvidenceObligation.
    """
    need_id:str
    text:str
    normalized:str
    created_step:int
    target:str=''
    question_type:str=''
    evidence_goal:str=''
    attempts:int=0
    evidence_ids:list[str]=field(default_factory=list)
    modes:list[str]=field(default_factory=list)
    no_gain_attempts:int=0
    lexical_no_gain_attempts:int=0
    exhausted:bool=False
    last_attempt_step:int|None=None
    last_gain_step:int|None=None
    match_quality:str='new'
    aliases:list[str]=field(default_factory=list)


_STOP={
    'the','a','an','to','of','in','for','and','or','is','are','was','were','be','been','being','does','do','did','that','this','which','what','where','how','when','why',
    'find','locate','identify','code','function','path','logic','implementation','current','exact','responsible','responsibility','from','with','by','into','given','could','might',
}


class InformationNeedTracker:
    """Track retrieval attempts without owning semantic resolution."""
    def __init__(self,max_no_gain_attempts:int=2):
        self.needs:dict[str,InformationNeed]={}; self.by_norm={}; self.max_no_gain_attempts=max(1,int(max_no_gain_attempts))

    @staticmethod
    def normalize(text:str)->str:
        text=(text or '').lower(); text=re.sub(r'[^a-z0-9_./\- ]+',' ',text); return ' '.join(text.split())

    @classmethod
    def _tokens(cls,text:str)->set[str]:
        return {x for x in cls.normalize(text).split() if len(x)>1 and x not in _STOP}

    @staticmethod
    def _jaccard(a:set[str],b:set[str])->float:
        if not a or not b: return 0.0
        return len(a & b)/len(a | b)

    @classmethod
    def _anchors(cls,text:str)->set[str]:
        n=cls.normalize(text)
        out=set(re.findall(r'[a-z0-9_./-]+\.py\b|\b__[a-z0-9_]+__\b|\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_.]*\b',n,re.I))
        return {x.lower() for x in out}

    def _match_score(self,need:InformationNeed,*,text:str,target:str,qtype:str,goal:str)->float:
        if qtype and need.question_type and qtype != need.question_type:
            return 0.0
        target_s=self._jaccard(self._tokens(target),self._tokens(need.target))
        goal_s=self._jaccard(self._tokens(goal),self._tokens(need.evidence_goal))
        text_s=self._jaccard(self._tokens(text),self._tokens(need.text+' '+' '.join(need.aliases)))
        anchors=self._anchors(text+' '+target+' '+goal)
        old_anchors=self._anchors(need.text+' '+need.target+' '+need.evidence_goal)
        anchor_bonus=0.20 if anchors and old_anchors and anchors & old_anchors else 0.0
        if target or goal:
            score=max(0.45*target_s+0.45*goal_s+0.10*text_s,0.65*text_s+0.35*max(target_s,goal_s))
        else:
            score=text_s
        return min(1.0,score+anchor_bonus)

    def get_or_create(self,text:str,step:int,structured:dict|None=None)->InformationNeed|None:
        n=self.normalize(text); structured=structured or {}
        target=self.normalize(structured.get('target') or '')
        qtype=self.normalize(structured.get('question_type') or '')
        goal=self.normalize(structured.get('evidence_goal') or '')
        if not n and not target and not goal:return None
        if n and n in self.by_norm:
            obj=self.needs[self.by_norm[n]]; obj.match_quality='exact_text'; return obj
        for obj in self.needs.values():
            if qtype and obj.question_type==qtype and target and goal and obj.target==target and obj.evidence_goal==goal:
                obj.match_quality='exact_structured'
                if n and n != obj.normalized and n not in obj.aliases: obj.aliases.append(n); self.by_norm[n]=obj.need_id
                return obj
        best=None; best_score=0.0
        for obj in self.needs.values():
            score=self._match_score(obj,text=n,target=target,qtype=qtype,goal=goal)
            if score>best_score: best=obj; best_score=score
        threshold=0.52 if (target or goal) else 0.62
        if best is not None and best_score>=threshold:
            best.match_quality=f'lexical_semantic:{best_score:.2f}'
            if n and n != best.normalized and n not in best.aliases: best.aliases.append(n); self.by_norm[n]=best.need_id
            if not best.target and target: best.target=target
            if not best.question_type and qtype: best.question_type=qtype
            if not best.evidence_goal and goal: best.evidence_goal=goal
            return best
        basis='|'.join([target,qtype,goal,n]) or n
        nid='N'+hashlib.sha1(basis.encode()).hexdigest()[:8]
        obj=InformationNeed(nid,(text or '').strip(),n,int(step),target,qtype,goal)
        obj.match_quality='structured_new' if (target or goal) else 'text_fallback_new'
        self.needs[nid]=obj
        if n:self.by_norm[n]=nid
        return obj

    def note_attempt(self,need:InformationNeed|None,mode:str='',step:int|None=None)->None:
        if not need:return
        need.attempts+=1; need.last_attempt_step=step
        if mode: need.modes.append(mode)
        # A new attempt can recover a previously exhausted retrieval path.
        if need.exhausted and need.no_gain_attempts < self.max_no_gain_attempts:
            need.exhausted=False

    def note_result(self,need:InformationNeed|None,evidence_ids:list[str],gained:bool,mode:str='',step:int|None=None)->None:
        if not need:return
        mode=mode or (need.modes[-1] if need.modes else '')
        for eid in evidence_ids:
            if eid not in need.evidence_ids: need.evidence_ids.append(eid)
        if gained:
            need.no_gain_attempts=0; need.exhausted=False; need.last_gain_step=step
            if mode in {'lexical','semantic','hybrid'}: need.lexical_no_gain_attempts=0
        else:
            need.no_gain_attempts+=1
            if mode=='lexical': need.lexical_no_gain_attempts+=1
            if mode in {'semantic','hybrid'}: need.lexical_no_gain_attempts=0
            if need.no_gain_attempts>=self.max_no_gain_attempts: need.exhausted=True

    def advisory(self,need:InformationNeed|None)->str:
        if not need:return ''
        if need.lexical_no_gain_attempts>=2:
            return 'This retrieval need had repeated lexical attempts without new evidence. Consider semantic or hybrid code_search rather than another lexical reformulation.'
        if need.exhausted:
            return 'This retrieval need has produced little new evidence after repeated attempts. Prefer validating the leading hypothesis or stop broad repeated search.'
        return ''

    def summary(self):
        return [asdict(x) for x in self.needs.values()]
