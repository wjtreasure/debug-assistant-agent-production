from __future__ import annotations
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from debug_assistant.models import ActionKind, ActionProposal, Evidence


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool = True
    hard_block: bool = False
    reason: str = ''
    advisory: str = ''


def _norm_path(value:str)->str:
    value=(value or '').replace('\\','/').strip()
    parts=[]
    for p in PurePosixPath(value).parts:
        if p in ('','.','/'): continue
        if p=='..':
            if parts: parts.pop()
            continue
        parts.append(p)
    return PurePosixPath(*parts).as_posix() if parts else '.'


def _source_dirs(evidence:Iterable[Evidence])->set[str]:
    dirs=set()
    for ev in evidence:
        if ev.source != 'read_file' or not ev.file:
            continue
        p=PurePosixPath(_norm_path(ev.file))
        parent=p.parent.as_posix()
        dirs.add(parent if parent not in ('','.') else '.')
    return dirs


def _within_scope(path:str,scopes:set[str])->bool:
    p=_norm_path(path)
    if p=='.': return False
    for scope in scopes:
        s=_norm_path(scope)
        if s=='.':
            # A root-level source file does not authorize a whole-repository tree walk.
            continue
        if p==s or p.startswith(s+'/') or s.startswith(p+'/'):
            return True
    return False


class ActionPolicy:
    """Phase-aware action-space governance.

    EXPLORE is permissive. CONVERGE is advisory-first with a narrow repo_tree
    exception. VERIFY_ONLY blocks broad exploration. FINALIZE/BUDGET_CRITICAL
    never starts a new investigation action.
    """
    def evaluate(self, action:ActionProposal, *, budget_phase:str, convergence_mode:str, evidence:list[Evidence])->PolicyDecision:
        if action.kind not in {ActionKind.TOOL,ActionKind.PARALLEL}:
            return PolicyDecision()
        if action.kind is ActionKind.PARALLEL:
            phase=(budget_phase or 'explore').lower(); mode=(convergence_mode or 'normal').lower()
            if phase=='finalize' or mode in {'budget_critical','force_finalization'}:
                return PolicyDecision(False,True,'finalization phase forbids starting a parallel investigation action')
            for child in action.actions:
                tool=str(child.get('tool') or ''); args=dict(child.get('arguments') or {})
                if phase=='verify_only':
                    if tool=='repo_tree':
                        return PolicyDecision(False,True,'verify_only forbids repository-wide structure exploration inside parallel group')
                    if tool=='code_search' and str(args.get('mode','lexical')).lower() in {'semantic','hybrid'}:
                        return PolicyDecision(False,True,'verify_only forbids broad semantic/hybrid retrieval inside parallel group')
                if (phase=='converge' or mode=='convergence_required') and tool=='repo_tree':
                    return PolicyDecision(False,True,'converge forbids broad repo_tree inside parallel group')
            advisory='VERIFY_ONLY: exact bounded parallel validation only.' if phase=='verify_only' else ('CONVERGE: bounded parallel reads must close current evidence gaps.' if phase=='converge' or mode=='convergence_required' else '')
            return PolicyDecision(True,False,'',advisory)
        phase=(budget_phase or 'explore').lower()
        mode=(convergence_mode or 'normal').lower()
        tool=action.tool or ''

        if phase=='finalize' or mode in {'budget_critical','force_finalization'}:
            return PolicyDecision(False,True,'finalization phase forbids starting a new investigation tool')

        if phase=='verify_only':
            if tool=='repo_tree':
                return PolicyDecision(False,True,'verify_only forbids repository-wide structure exploration')
            if tool=='code_search' and str(action.arguments.get('mode','lexical')).lower() in {'semantic','hybrid'}:
                return PolicyDecision(False,True,'verify_only forbids broad semantic/hybrid retrieval; use exact source validation')
            return PolicyDecision(True,False,'','VERIFY_ONLY: restrict work to exact validation of the current hypothesis or critical evidence gap.')

        if phase=='converge' or mode=='convergence_required':
            if tool=='repo_tree':
                path=str(action.arguments.get('path','.') or '.')
                depth=int(action.arguments.get('depth',1) or 1)
                max_entries=int(action.arguments.get('max_entries',100) or 100)
                scopes=_source_dirs(evidence)
                if not (_within_scope(path,scopes) and depth<=2 and max_entries<=100):
                    return PolicyDecision(False,True,'converge permits repo_tree only for a narrow directory already grounded by source evidence')
                return PolicyDecision(True,False,'','CONVERGE: scoped repo_tree is allowed only as local structure confirmation; prefer exact read/search.')
            return PolicyDecision(True,False,'','CONVERGE: stay inside the current root-cause scope, close existing evidence obligations, and avoid opening unrelated investigation branches.')

        return PolicyDecision()
