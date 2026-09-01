"""Parse and summarize the kidney-exchange XML instances used in the study."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DonorRecord:
    """Lightweight container for donor level data extracted from an instance."""

    donor_id: int
    bloodtype: str
    source_patient_ids: tuple[int, ...]
    matches: tuple[tuple[int, float], ...]


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
