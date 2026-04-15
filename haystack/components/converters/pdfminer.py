# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pdfminer.six'") as pdfminer_import:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer

logger = logging.getLogger(__name__)

CID_PATTERN = r"\(cid:\d+\)"


@component
class PDFMinerToDocument:
    """
    Converts PDF files to Documents using pdfminer.
    """

    def __init__(
        self,
        line_overlap: float = 0.5,
        char_margin: float = 2.0,
        line_margin: float = 0.5,
        word_margin: float = 0.1,
        boxes_flow: float | None = 0.5,
        detect_vertical: bool = True,
        all_texts: bool = False,
        store_full_path: bool = False,
    ) -> None:
        pdfminer_import.check()

        self.layout_params = LAParams(
            line_overlap=line_overlap,
            char_margin=char_margin,
            line_margin=line_margin,
            word_margin=word_margin,
            boxes_flow=boxes_flow,
            detect_vertical=detect_vertical,
            all_texts=all_texts,
        )

        self.store_full_path = store_full_path
        self.cid_pattern = re.compile(CID_PATTERN)

    @staticmethod
    def _converter(lt_page_objs: Iterator) -> str:
        """
        Convert PDF pages to a single string.
        """
        pages: list[str] = []

        for page in lt_page_objs:
            page_chunks: list[str] = []

            for container in page:
                if isinstance(container, LTTextContainer):
                    text = container.get_text().strip()
                    if text:
                        page_chunks.append(text)

            pages.append("\n\n".join(page_chunks))

        return "\f".join(pages)

    def detect_undecoded_cid_characters(self, text: str) -> dict[str, Any]:
        """
        Detect CID (undecoded) characters in extracted text.
        """
        matches = self.cid_pattern.findall(text)

        total_chars = len(text)
        cid_chars = sum(len(match) for match in matches)

        percentage = (cid_chars / total_chars * 100) if total_chars else 0

        return {
            "total_chars": total_chars,
            "cid_chars": cid_chars,
            "percentage": round(percentage, 2),
        }

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Convert PDF sources into Haystack Documents.
        """
        documents: list[Document] = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    "Skipping source {source}: failed to read ({error})",
                    source=source,
                    error=e,
                )
                continue

            try:
                pages = extract_pages(
                    io.BytesIO(bytestream.data),
                    laparams=self.layout_params,
                )
                text = self._converter(pages)
            except Exception as e:
                logger.warning(
                    "Skipping source {source}: extraction failed ({error})",
                    source=source,
                    error=e,
                )
                continue

            if not text or not text.strip():
                logger.warning(
                    "Empty text extracted from {source}",
                    source=source,
                )

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path:
                file_path = bytestream.meta.get("file_path")
                if file_path:
                    merged_metadata["file_path"] = os.path.basename(file_path)

            # CID detection
            analysis = self.detect_undecoded_cid_characters(text)

            if analysis["percentage"] > 0:
                logger.warning(
                    "CID issue in {source}: {cid_chars}/{total_chars} chars ({percentage}%)",
                    source=source,
                    cid_chars=analysis["cid_chars"],
                    total_chars=analysis["total_chars"],
                    percentage=analysis["percentage"],
                )

            documents.append(
                Document(
                    content=text,
                    meta=merged_metadata,
                )
            )

        return {"documents": documents}