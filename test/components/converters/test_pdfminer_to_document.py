# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import patch

import pytest

from haystack import Document
from haystack.components.converters.pdfminer import PDFMinerToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ByteStream


class TestPDFMinerToDocument:
    def test_run(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content

    def test_init_params_custom(self, test_files_path):
        converter = PDFMinerToDocument(char_margin=0.5, all_texts=True, store_full_path=False)
        assert converter.layout_params.char_margin == 0.5
        assert converter.layout_params.all_texts is True
        assert converter.store_full_path is False

    def test_run_with_store_full_path_false(self, test_files_path):
        converter = PDFMinerToDocument(store_full_path=False)
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content
            assert doc.meta["file_path"] == "sample_pdf_1.pdf"

    def test_run_wrong_file_type(self, test_files_path, caplog):
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = PDFMinerToDocument()

        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=sources)
            assert "Is this really a PDF?" in caplog.text

        assert not output["documents"]

    def test_arg_is_none(self, test_files_path):
        converter = PDFMinerToDocument(char_margin=None)
        assert converter.layout_params.char_margin is None

    def test_run_doc_metadata(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        metadata = [{"file_name": "sample_pdf_2.pdf"}]

        results = converter.run(sources=sources, meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta["file_name"] == "sample_pdf_2.pdf"

    def test_incorrect_meta(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_3.pdf"]
        metadata = [{"file_name": "sample_pdf_3.pdf"}, {"file_name": "sample_pdf_2.pdf"}]

        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
            converter.run(sources=sources, meta=metadata)

    def test_run_bytestream_metadata(self, test_files_path):
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            stream = ByteStream(file.read(), meta={"content_type": "text/pdf", "url": "test_url"})

        results = converter.run(sources=[stream])
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {"content_type": "text/pdf", "url": "test_url"}

    def test_run_bytestream_doc_overlapping_metadata(self, test_files_path):
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            stream = ByteStream(file.read(), meta={"content_type": "text/pdf", "url": "test_url_correct"})

        metadata = [{"file_name": "sample_pdf_2.pdf", "url": "test_url_new"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {
            "file_name": "sample_pdf_2.pdf",
            "content_type": "text/pdf",
            "url": "test_url_new",
        }

    def test_run_error_handling(self, caplog):
        converter = PDFMinerToDocument()

        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=["non_existing_file.pdf"])
            assert "Could not read non_existing_file.pdf" in caplog.text

        assert results["documents"] == []

    def test_run_empty_document(self, caplog, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "non_text_searchable.pdf"]

        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "PDFMinerToDocument could not extract text from the file" in caplog.text

        doc = results["documents"][0]
        assert doc.content == ""
        assert doc.meta["file_path"] == "non_text_searchable.pdf"
        assert doc.id != Document(content="").id

    def test_run_detect_pages_and_split_by_passage(self, test_files_path):
        converter = PDFMinerToDocument()
        pdf_doc = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_2.pdf"])

        splitter = DocumentSplitter(split_length=1, split_by="page")
        docs = splitter.run(pdf_doc["documents"])

        assert len(docs["documents"]) == 4

    def test_run_detect_paragraphs_to_be_used_in_split_passage(self, test_files_path):
        converter = PDFMinerToDocument()
        pdf_doc = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_2.pdf"])

        splitter = DocumentSplitter(split_length=1, split_by="passage")
        docs = splitter.run(pdf_doc["documents"])

        assert len(docs["documents"]) == 29

    # ---------------- NEW TESTS ----------------

    def test_link_format_none(self, monkeypatch):
        converter = PDFMinerToDocument(link_format="none")

        def mock_converter(*args, **kwargs):
            return "Example link text"

        with patch.object(PDFMinerToDocument, "_converter", side_effect=mock_converter):
            result = converter.run(sources=[ByteStream(b"data")])
            assert "http" not in result["documents"][0].content

    def test_link_format_markdown(self, monkeypatch):
        converter = PDFMinerToDocument(link_format="markdown")

        def mock_converter(*args, **kwargs):
            return "[Google](https://google.com)"

        with patch.object(PDFMinerToDocument, "_converter", side_effect=mock_converter):
            result = converter.run(sources=[ByteStream(b"data")])
            assert "[Google](https://google.com)" in result["documents"][0].content

    def test_link_format_plain(self, monkeypatch):
        converter = PDFMinerToDocument(link_format="plain")

        def mock_converter(*args, **kwargs):
            return "Google (https://google.com)"

        with patch.object(PDFMinerToDocument, "_converter", side_effect=mock_converter):
            result = converter.run(sources=[ByteStream(b"data")])
            assert "Google (https://google.com)" in result["documents"][0].content

    def test_to_dict_from_dict_link_format(self):
        converter = PDFMinerToDocument(link_format="markdown")
        data = converter.to_dict()

        new_converter = PDFMinerToDocument.from_dict(data)
        assert new_converter.link_format == "markdown"

    def test_pdf_with_no_annotations(self, monkeypatch):
        converter = PDFMinerToDocument()

        def mock_extract_pages(*args, **kwargs):
            return ["page"]

        def mock_converter(*args, **kwargs):
            return "Text without annotations"

        with patch("haystack.components.converters.pdfminer.extract_pages", side_effect=mock_extract_pages):
            with patch.object(PDFMinerToDocument, "_converter", side_effect=mock_converter):
                result = converter.run(sources=[ByteStream(b"data")])
                assert result["documents"][0].content == "Text without annotations"