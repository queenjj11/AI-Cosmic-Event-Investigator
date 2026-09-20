"""Astrophysical document chunking preserving taxonomy headers and diagnostic criteria."""

from typing import Any, Dict, List
from dataclasses import dataclass, field
import re

from src.knowledge_base.document_loader import AstroDocument


@dataclass
class DocumentChunk:
    """A searchable chunk of astronomical text with source provenance."""
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_type: str
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Splits astronomical research papers and catalog entries into searchable chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: AstroDocument) -> List[DocumentChunk]:
        """Split document by markdown sections or character windows."""
        # For catalogs, keep each entry as a coherent chunk
        if doc.source_type == "catalog":
            return [DocumentChunk(
                chunk_id=f"{doc.doc_id}_chunk_0",
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.content.strip(),
                source_type=doc.source_type,
                filepath=doc.filepath,
                metadata=doc.metadata
            )]

        # For papers, split primarily by major section headers
        sections = re.split(r"\n(?=###?\s)", doc.content)
        chunks: List[DocumentChunk] = []

        for i, sec in enumerate(sections):
            text = sec.strip()
            if not text:
                continue

            if len(text) <= self.chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc.doc_id}_sec_{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    source_type=doc.source_type,
                    filepath=doc.filepath,
                    metadata=doc.metadata
                ))
            else:
                # Sub-chunk with overlap
                start = 0
                sub_idx = 0
                while start < len(text):
                    end = start + self.chunk_size
                    sub_text = text[start:end]
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc.doc_id}_sec_{i}_sub_{sub_idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=sub_text.strip(),
                        source_type=doc.source_type,
                        filepath=doc.filepath,
                        metadata=doc.metadata
                    ))
                    start += self.chunk_size - self.chunk_overlap
                    sub_idx += 1

        return chunks

    def chunk_all(self, docs: List[AstroDocument]) -> List[DocumentChunk]:
        """Chunk a list of documents."""
        all_chunks = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        return all_chunks
