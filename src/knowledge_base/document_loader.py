"""Loads astrophysical literature, preprints, and transient catalogs."""

import os
import json
from typing import Any, Dict, List
from dataclasses import dataclass, field


@dataclass
class AstroDocument:
    """Document containing astronomical research or catalog reference."""
    doc_id: str
    title: str
    content: str
    source_type: str  # "paper" or "catalog"
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads all scientific papers and catalogs from the knowledge base directory."""

    def __init__(self, papers_dir: str = "knowledge_base/papers",
                 catalogs_dir: str = "knowledge_base/catalogs"):
        self.papers_dir = papers_dir
        self.catalogs_dir = catalogs_dir

    def load_all(self) -> List[AstroDocument]:
        """Load all papers and catalogs found in the knowledge base directories."""
        documents: List[AstroDocument] = []

        # 1. Load papers (.md, .txt)
        if os.path.exists(self.papers_dir):
            for fname in sorted(os.listdir(self.papers_dir)):
                fpath = os.path.join(self.papers_dir, fname)
                if fname.endswith((".md", ".txt")):
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    lines = text.strip().split("\n")
                    title = lines[0].replace("#", "").strip() if lines else fname
                    documents.append(AstroDocument(
                        doc_id=fname,
                        title=title,
                        content=text,
                        source_type="paper",
                        filepath=fpath,
                        metadata={"filename": fname}
                    ))

        # 2. Load catalogs (.json)
        if os.path.exists(self.catalogs_dir):
            for fname in sorted(os.listdir(self.catalogs_dir)):
                fpath = os.path.join(self.catalogs_dir, fname)
                if fname.endswith(".json"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    cat_name = data.get("catalog_name", fname)
                    entries = data.get("entries", [])
                    for i, entry in enumerate(entries):
                        cls_name = entry.get("canonical_name", entry.get("class_id", f"Class_{i}"))
                        # Format entry as readable text block
                        content = f"CATALOG REFERENCE: {cls_name} ({entry.get('class_id')})\n"
                        content += f"Physical Mechanism: {entry.get('physical_mechanism', 'N/A')}\n"
                        content += f"Peak Absolute Magnitude Range: {entry.get('peak_absolute_magnitude_range', 'N/A')}\n"
                        content += f"Rise Time: {entry.get('rise_time_days', 'N/A')} days\n"
                        content += f"Decay Rate: {entry.get('decay_rate_mag_per_day', 'N/A')} mag/day\n"
                        content += f"Peak Color (g - r): {entry.get('color_g_minus_r_at_peak', 'N/A')}\n"
                        content += f"Diagnostic Signatures: {entry.get('diagnostic_signatures', 'N/A')}\n"
                        content += f"Literature Reference: {entry.get('literature_reference', 'N/A')}\n"

                        documents.append(AstroDocument(
                            doc_id=f"{fname}_{entry.get('class_id', i)}",
                            title=f"Catalog: {cls_name}",
                            content=content,
                            source_type="catalog",
                            filepath=fpath,
                            metadata=entry
                        ))

        return documents
