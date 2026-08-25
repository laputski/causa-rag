"""A structure parser for a technical manual.

A manual numbers its sections: "4.2.1 Replacing the detector". The number is
the path, and it already says that the section nests inside 4.2, which nests
inside 4. Without a parser the chunker cuts the file with a character window and
a fragment arrives with no structural path at all: "no structure" on screen, and
nothing to check a source reference against in the metrics.

That is the pack's job: the platform does not know what a heading looks like in
a particular subject area, and it should not have to.
"""
from __future__ import annotations

import re

from core.models import DocumentNode

# "4.2.1 Replacing the detector", "7 Safety", "A.1 Appendix". The number takes
# Latin or Cyrillic letters as well as digits, so a manual in either script
# parses. The dot after the number is optional, because manuals write it both
# ways.
_HEADING = re.compile(r"^\s*((?:\d+|[A-ZА-Я])(?:\.\d+)*)\.?\s+(\S.*)$")


def parse_manual_section(content: str) -> DocumentNode:
    """Build a tree of sections from numbered headings.

    Depth comes from the number itself and not from indentation, because
    indentation does not survive conversion out of a PDF and a number does.
    """
    root = DocumentNode(node_id="root", node_type="document", level=0)
    stack: list[DocumentNode] = [root]

    for raw in content.splitlines():
        match = _HEADING.match(raw)
        if match is None:
            if len(stack) > 1:
                node = stack[-1]
                node.content = f"{node.content}\n{raw}".strip() if node.content else raw.strip()
            continue
        number, title = match.group(1), match.group(2).strip()
        level = number.count(".") + 1
        node = DocumentNode(
            node_id=number,
            node_type="section",
            title=title,
            level=level,
            metadata={"section_no": number},
        )
        # A deeper heading nests; one at the same level or above sits beside
        # the nearest ancestor whose level is lower.
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)

    return root
