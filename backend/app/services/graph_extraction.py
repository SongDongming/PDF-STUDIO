"""Deterministic merge for batched, PDF-grounded graph extractions."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _entity_key(kind: str, name: str) -> tuple[str, str]:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    normalized = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    if not normalized:
        raise ValueError("graph extraction entity name is not normalizable")
    return kind, normalized


def merge_graph_extractions(
    document_id: str, batches: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge batch-local refs and evidence into one strict compiler envelope."""

    entities: dict[tuple[str, str], dict[str, Any]] = {}
    claims: dict[tuple[str, str], dict[str, Any]] = {}
    relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for batch_index, batch in enumerate(batches, start=1):
        if str(batch.get("schema_version")) != "1.0":
            raise ValueError(f"graph extraction batch {batch_index} has invalid schema")
        if str(batch.get("document_id")) != document_id:
            raise ValueError(
                f"graph extraction batch {batch_index} changed document lineage"
            )
        raw_entities = batch.get("entities")
        raw_claims = batch.get("claims")
        raw_relations = batch.get("relations")
        if not all(
            isinstance(value, list)
            for value in (raw_entities, raw_claims, raw_relations)
        ):
            raise ValueError(
                f"graph extraction batch {batch_index} has invalid arrays"
            )

        local_refs: dict[str, tuple[str, str]] = {}
        for raw in raw_entities:
            if not isinstance(raw, Mapping):
                raise ValueError("graph extraction entity must be an object")
            ref = str(raw.get("ref") or "")
            name = str(raw.get("name") or "").strip()
            kind = str(raw.get("kind") or "").strip()
            if not ref or not name or not kind:
                raise ValueError("graph extraction entity is incomplete")
            key = _entity_key(kind, name)
            local_refs[ref] = key
            existing = entities.get(key)
            aliases = {
                str(value).strip()
                for value in raw.get("aliases", [])
                if str(value).strip()
            }
            evidence = {
                str(value).strip()
                for value in raw.get("evidence_refs", [])
                if str(value).strip()
            }
            if existing is None:
                entities[key] = {
                    "name": name,
                    "kind": kind,
                    "aliases": aliases,
                    "evidence_refs": evidence,
                }
            else:
                existing["aliases"].update(aliases)
                existing["evidence_refs"].update(evidence)

        for raw in raw_claims:
            if not isinstance(raw, Mapping):
                raise ValueError("graph extraction claim must be an object")
            subject = local_refs.get(str(raw.get("subject_ref") or ""))
            if subject is None:
                raise ValueError("graph extraction claim references unknown entity")
            statement = str(raw.get("statement") or "").strip()
            key = (repr(subject), statement)
            evidence = {
                str(value).strip()
                for value in raw.get("evidence_refs", [])
                if str(value).strip()
            }
            if key not in claims:
                claims[key] = {
                    "subject": subject,
                    "statement": statement,
                    "evidence_refs": evidence,
                }
            else:
                claims[key]["evidence_refs"].update(evidence)

        for raw in raw_relations:
            if not isinstance(raw, Mapping):
                raise ValueError("graph extraction relation must be an object")
            source = local_refs.get(str(raw.get("source_ref") or ""))
            target = local_refs.get(str(raw.get("target_ref") or ""))
            if source is None or target is None:
                raise ValueError("graph extraction relation references unknown entity")
            predicate = str(raw.get("predicate") or "").strip()
            statement = str(raw.get("statement") or "").strip()
            key = (repr(source), repr(target), predicate, statement)
            evidence = {
                str(value).strip()
                for value in raw.get("evidence_refs", [])
                if str(value).strip()
            }
            if key not in relations:
                relations[key] = {
                    "source": source,
                    "target": target,
                    "predicate": predicate,
                    "statement": statement,
                    "evidence_refs": evidence,
                }
            else:
                relations[key]["evidence_refs"].update(evidence)

    ordered_entities = sorted(
        entities.items(), key=lambda item: (item[0][0], item[0][1])
    )
    refs = {
        key: f"entity_{index:04d}"
        for index, (key, _) in enumerate(ordered_entities, start=1)
    }
    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "entities": [
            {
                "ref": refs[key],
                "name": value["name"],
                "kind": value["kind"],
                "aliases": sorted(value["aliases"]),
                "evidence_refs": sorted(value["evidence_refs"]),
            }
            for key, value in ordered_entities
        ],
        "claims": [
            {
                "ref": f"claim_{index:04d}",
                "subject_ref": refs[value["subject"]],
                "statement": value["statement"],
                "evidence_refs": sorted(value["evidence_refs"]),
            }
            for index, value in enumerate(
                (claims[key] for key in sorted(claims)), start=1
            )
        ],
        "relations": [
            {
                "ref": f"relation_{index:04d}",
                "source_ref": refs[value["source"]],
                "target_ref": refs[value["target"]],
                "predicate": value["predicate"],
                "statement": value["statement"],
                "evidence_refs": sorted(value["evidence_refs"]),
            }
            for index, value in enumerate(
                (relations[key] for key in sorted(relations)), start=1
            )
        ],
    }
