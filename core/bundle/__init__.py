"""The fix bundle.

The platform does not apply corrections. It publishes a versioned,
self-contained artefact that a served system loads at startup, holds in
memory and applies locally, never reaching back to the platform. That is the
only shape compatible with this platform standing outside the hot path, and
it is also what industrial search engines do:
query rules live inside the engine, not in an external store it consults per
request.

Three decisions are load-bearing here.

**A portable anchor instead of a chunk id.** A chunk id in this platform is
position-derived, so it means a different piece of text after re-indexing.
An anchor names content — document, structural path, hash of the normalised
text, and the text itself — and the recipient resolves it into *its own*
identifiers at load time, not per query. Unresolved anchors are reported
rather than skipped.

**Signature text instead of a vector.** An embedding belongs to one model,
so shipping vectors would expire the artefact the day the recipient changes
models. The recipient embeds the text with its own model, and the bundle
carries a calibration procedure rather than a threshold number.

**A minimal format.** No signatures, no delta updates, no version
negotiation until a second accepting recipient exists. Every one of those is
easy to add later and impossible to remove once shipped.
"""
from core.bundle.anchor import Anchor, AnchorResolution, resolve_anchors
from core.bundle.bundle import FixBundle, FixEntry, bundle_from_judgments, load_bundle

__all__ = [
    "Anchor",
    "AnchorResolution",
    "resolve_anchors",
    "FixBundle",
    "FixEntry",
    "bundle_from_judgments",
    "load_bundle",
]
