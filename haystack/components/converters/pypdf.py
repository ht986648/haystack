# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
import os
from enum import Enum
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdf'") as pypdf_import:
    from pypdf import PdfReader


logger = logging.getLogger(__name__)


class PyPDFExtractionMode(Enum):
    PLAIN = "plain"
    LAYOUT = "layout"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "PyPDFExtractionMode":
        enum_map = {e.value: e for e in PyPDFExtractionMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown extraction mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


class PDFLinkFormat(Enum):
    """Supported formats for link extraction from PDF files."""
    MARKDOWN = "markdown"
    PLAIN = "plain"
    NONE = "none"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "PDFLinkFormat":
        enum_map = {e.value: e for e in PDFLinkFormat}
        fmt = enum_map.get(string.lower())
        if fmt is None:
            raise ValueError(f"Unknown link format '{string}'. Supported: {list(enum_map.keys())}")
        return fmt


@component
class PyPDFToDocument:
    def __init__(
        self,
        *,
        extraction_mode: str | PyPDFExtractionMode = PyPDFExtractionMode.PLAIN,
        plain_mode_orientations: tuple = (0, 90, 180, 270),
        plain_mode_space_width: float = 200.0,
        layout_mode_space_vertically: bool = True,
        layout_mode_scale_weight: float = 1.25,
        layout_mode_strip_rotated: bool = True,
        layout_mode_font_height_weight: float = 1.0,
        store_full_path: bool = False,
        link_format: str | PDFLinkFormat = PDFLinkFormat.NONE,   # ✅ NEW
    ) -> None:
        pypdf_import.check()

        self.store_full_path = store_full_path

        if isinstance(extraction_mode, str):
            extraction_mode = PyPDFExtractionMode.from_str(extraction_mode)
        self.extraction_mode = extraction_mode

        if isinstance(link_format, str):
            link_format = PDFLinkFormat.from_str(link_format)
        self.link_format = link_format  # ✅ NEW

        self.plain_mode_orientations = plain_mode_orientations
        self.plain_mode_space_width = plain_mode_space_width
        self.layout_mode_space_vertically = layout_mode_space_vertically
        self.layout_mode_scale_weight = layout_mode_scale_weight
        self.layout_mode_strip_rotated = layout_mode_strip_rotated
        self.layout_mode_font_height_weight = layout_mode_font_height_weight

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            extraction_mode=str(self.extraction_mode),
            link_format=str(self.link_format),  # ✅ NEW
            plain_mode_orientations=self.plain_mode_orientations,
            plain_mode_space_width=self.plain_mode_space_width,
            layout_mode_space_vertically=self.layout_mode_space_vertically,
            layout_mode_scale_weight=self.layout_mode_scale_weight,
            layout_mode_strip_rotated=self.layout_mode_strip_rotated,
            layout_mode_font_height_weight=self.layout_mode_font_height_weight,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PyPDFToDocument":
        return default_from_dict(cls, data)

    def _extract_links_from_page(self, page) -> list[tuple[str, str]]:
        """Returns list of (text/rect_label, url) tuples from page annotations."""
        links = []
        if "/Annots" not in page:
            return links

        for annot in page["/Annots"]:
            try:
                obj = annot.get_object()
                if obj.get("/Subtype") == "/Link":
                    action = obj.get("/A", {})
                    url = action.get("/URI", "")
                    if url:
                        display = obj.get("/Contents", url)
                        links.append((str(display), str(url)))
            except Exception as e:
                logger.debug("Error extracting link: {error}", error=e)

        return links

    def _default_convert(self, reader: "PdfReader") -> str:
        texts = []
        for page in reader.pages:
            extracted_text = page.extract_text(
                orientations=self.plain_mode_orientations,
                extraction_mode=self.extraction_mode.value,
                space_width=self.plain_mode_space_width,
                layout_mode_space_vertically=self.layout_mode_space_vertically,
                layout_mode_scale_weight=self.layout_mode_scale_weight,
                layout_mode_strip_rotated=self.layout_mode_strip_rotated,
                layout_mode_font_height_weight=self.layout_mode_font_height_weight,
            )

            text = extracted_text or ""

            # ✅ Inject links
            if self.link_format != PDFLinkFormat.NONE:
                links = self._extract_links_from_page(page)
                for display, url in links:
                    if self.link_format == PDFLinkFormat.MARKDOWN:
                        text += f"\n[{display}]({url})"
                    else:  # PLAIN
                        text += f"\n{display} ({url})"

            texts.append(text)

        return "\f".join(texts)

    @component.output_types(documents=list[Document])
    def run(
        self, sources: list[str | Path | ByteStream], meta: dict[str, Any] | list[dict[str, Any]] | None = None
    ) -> dict[str, list[Document]]:
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            try:
                pdf_reader = PdfReader(io.BytesIO(bytestream.data))
                text = self._default_convert(pdf_reader)
            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}",
                    source=source,
                    error=e,
                )
                continue

            if text is None or text.strip() == "":
                logger.warning(
                    "PyPDFToDocument could not extract text from the file {source}. Returning an empty document.",
                    source=source,
                )

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (file_path := bytestream.meta.get("file_path")):
                merged_metadata["file_path"] = os.path.basename(file_path)

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}