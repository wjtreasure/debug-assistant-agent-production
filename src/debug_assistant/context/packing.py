from __future__ import annotations


def line_safe_truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Bound text at complete-line boundaries whenever the input permits it."""
    max_chars=max(0,int(max_chars))
    if len(text)<=max_chars:
        return text,False
    kept=[]; used=0
    for line in text.splitlines():
        addition=len(line)+(1 if kept else 0)
        if used+addition>max_chars:
            break
        kept.append(line); used+=addition
    if kept:
        return "\n".join(kept),True
    return text[:max_chars],True


def line_safe_head_tail(text: str, max_chars: int, *, marker: str="...[middle omitted]...") -> str:
    """Keep bounded complete lines from both ends of a diagnostic payload."""
    max_chars=max(0,int(max_chars))
    if len(text)<=max_chars:
        return text
    if max_chars<=len(marker)+2:
        return marker[:max_chars]
    lines=text.splitlines()
    each=max(0,(max_chars-len(marker)-2)//2)
    head,_=line_safe_truncate(text,each)
    tail,_=line_safe_truncate("\n".join(reversed(lines)),each)
    tail="\n".join(reversed(tail.splitlines()))
    return f"{head}\n{marker}\n{tail}"
