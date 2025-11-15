from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class DonorRecord:
    """Lightweight container for donor level data extracted from an instance."""

    donor_id: int
    bloodtype: str
    source_patient_ids: tuple[int, ...]
    matches: tuple[tuple[int, float], ...]

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    @property
    def is_altruistic(self) -> bool:
        return len(self.source_patient_ids) == 0


@dataclass(frozen=True)
class RecipientRecord:
    """Container for recipient level data extracted from an instance."""

    recipient_id: int
    bloodtype: str
    c_pra: float | None
    has_blood_compatible_donor: bool | None


@dataclass(frozen=True)
class InstanceData:
    """Holds the parsed content of an instance file."""

    name: str
    donors: tuple[DonorRecord, ...]
    recipients: tuple[RecipientRecord, ...]


def _parse_match(match_element: ET.Element) -> tuple[int, float]:
    recipient_id = int(match_element.findtext("recipient"))
    score_text = match_element.findtext("score")
    score = float(score_text) if score_text is not None else float("nan")
    return recipient_id, score


def _parse_donor(entry: ET.Element) -> DonorRecord:
    matches_element = entry.find("matches")
    sources_element = entry.find("sources")
    matches = (
        tuple(_parse_match(match) for match in matches_element.findall("match"))
        if matches_element is not None
        else tuple()
    )
    sources = (
        tuple(int(source.text) for source in sources_element.findall("source"))
        if sources_element is not None
        else tuple()
    )
    return DonorRecord(
        donor_id=int(entry.get("donor_id")),
        bloodtype=entry.get("bloodtype", ""),
        source_patient_ids=sources,
        matches=matches,
    )


def _parse_recipient(recipient_element: ET.Element) -> RecipientRecord:
    c_pra_text = recipient_element.get("cPRA")
    has_compatible_text = recipient_element.get("hasBloodCompatibleDonor")
    return RecipientRecord(
        recipient_id=int(recipient_element.get("recip_id")),
        bloodtype=recipient_element.get("bloodtype", ""),
        c_pra=float(c_pra_text) if c_pra_text is not None else None,
        has_blood_compatible_donor=(
            has_compatible_text.lower() == "true"
            if has_compatible_text is not None
            else None
        ),
    )


def load_instance(path: str | Path) -> InstanceData:
    """
    Parse an XML instance file and return structured donor/recipient data.
    """
    xml_path = Path(path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    donor_records = tuple(_parse_donor(entry) for entry in root.findall("entry"))
    recipients_element = root.find("recipients")
    recipient_records = (
        tuple(
            _parse_recipient(recipient_element)
            for recipient_element in recipients_element.findall("recipient")
        )
        if recipients_element is not None
        else tuple()
    )

    return InstanceData(
        name=xml_path.name,
        donors=donor_records,
        recipients=recipient_records,
    )


def summarize_instance(instance: InstanceData) -> dict[str, Any]:
    """
    Produce high-level statistics for a parsed instance.
    """
    donors = instance.donors
    recipients = instance.recipients

    altruistic_donors = [donor for donor in donors if donor.is_altruistic]
    paired_donors = [donor for donor in donors if not donor.is_altruistic]
    donors_without_matches = [donor for donor in donors if donor.num_matches == 0]

    total_matches = sum(donor.num_matches for donor in donors)
    match_scores: list[float] = [
        score for donor in donors for (_, score) in donor.matches if score == score
    ]

    def _avg_matches(items: Iterable[DonorRecord]) -> float | None:
        counts = [donor.num_matches for donor in items]
        return mean(counts) if counts else None

    summary: dict[str, Any] = {
        "instance": instance.name,
        "donors_total": len(donors),
        "donors_altruistic": len(altruistic_donors),
        "donors_paired": len(paired_donors),
        "donors_without_matches": len(donors_without_matches),
        "recipients_total": len(recipients),
        "matches_total": total_matches,
        "avg_matches_per_donor": total_matches / len(donors) if donors else None,
        "avg_matches_altruistic": _avg_matches(altruistic_donors),
        "avg_matches_paired": _avg_matches(paired_donors),
        "avg_match_score": mean(match_scores) if match_scores else None,
        "donor_bloodtype_counts": dict(
            Counter(donor.bloodtype for donor in donors)
        ),
        "recipient_bloodtype_counts": dict(
            Counter(recipient.bloodtype for recipient in recipients)
        ),
        "altruistic_donor_ids": [donor.donor_id for donor in altruistic_donors],
        "donor_ids_without_matches": [donor.donor_id for donor in donors_without_matches],
    }
    return summary


def summarize_instances(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """
    Convenience helper that loads and summarizes many instance files.
    """
    summaries = []
    for path in paths:
        instance = load_instance(path)
        summaries.append(summarize_instance(instance))
    return summaries

