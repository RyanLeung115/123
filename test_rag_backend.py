"""Tests for rag_backend.py.  Run:  python test_rag_backend.py

The default tests need no Ollama, no PDFs and no vector store: they cover course-code
detection, chunking, the never-raise contract and the regression that made the first index
useless (the form header "Jul 2025" was matched as a course code, so 46 of 70 chunks were
filed under JUL2025/JUL2022 instead of COMP4431/COMP4432).

End-to-end tests that really embed + query need the dependencies and knowledge_base/:
    $env:RAG_E2E="1"; python test_rag_backend.py
"""
import os
import unittest

import rag_backend as rag


class TestCodeDetection(unittest.TestCase):
    def test_plain_code(self):
        self.assertEqual(rag.extract_course_code("COMP4431"), "COMP4431")

    def test_spaced_and_lowercase(self):
        self.assertEqual(rag.extract_course_code("tell me about comp 4431"), "COMP4431")
        self.assertEqual(rag.extract_course_code("COMP-4431"), "COMP4431")

    def test_three_letter_codes_are_kept(self):
        self.assertEqual(rag.extract_course_code("prerequisite AMA1007"), "AMA1007")

    def test_month_year_is_not_a_course(self):
        # Regression: the SD form header "Jul 2025" has the shape of a course code.
        for text in ["Jul 2025", "JULY2025", "AUG 2025", "Jul 2021"]:
            self.assertEqual(rag.extract_course_code(text), "", text)

    def test_declared_subject_code(self):
        self.assertEqual(rag.declared_subject_code("Subject Code COMP4432 Subject Title"), "COMP4432")
        self.assertEqual(rag.declared_subject_code("subject code: COMP4431"), "COMP4431")
        self.assertEqual(rag.declared_subject_code("see COMP2011 for details"), "")

    def test_bare_code(self):
        self.assertTrue(rag._is_bare_code("COMP4431"))
        self.assertTrue(rag._is_bare_code("comp 4431?"))
        self.assertFalse(rag._is_bare_code("what is the assessment of COMP4431"))


class TestChunking(unittest.TestCase):
    def test_long_page_is_split(self):
        chunks = rag.chunk_text("\n\n".join(["word " * 80] * 6))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= rag.CHUNK_SIZE * 1.5 + 1 for c in chunks), [len(c) for c in chunks])

    def test_tiny_fragment_is_dropped(self):
        self.assertEqual(rag.chunk_text("tiny"), [])

    def test_paragraphs_are_kept_together(self):
        chunks = rag.chunk_text("Objectives\n\n" + "x " * 200)
        self.assertTrue(chunks[0].startswith("Objectives"))


class TestContract(unittest.TestCase):
    """The public contract: always a dict, never an exception."""

    def assertEmpty(self, result):
        self.assertEqual(result, {"answer": "", "source_files": []})

    def test_empty_input(self):
        self.assertEmpty(rag.rag_course_query(""))
        self.assertEmpty(rag.rag_course_query(None))
        self.assertEmpty(rag.rag_course_query("   "))

    def test_no_index_available(self):
        original = rag._ensure_index
        rag._ensure_index = lambda: None
        try:
            self.assertEmpty(rag.rag_course_query("COMP4431"))
        finally:
            rag._ensure_index = original

    def test_internal_exception_is_swallowed(self):
        original = rag._ensure_index

        def boom():
            raise RuntimeError("vector store is broken")

        rag._ensure_index = boom
        try:
            self.assertEmpty(rag.rag_course_query("COMP4431"))
        finally:
            rag._ensure_index = original


@unittest.skipUnless(os.getenv("RAG_E2E") == "1", "set RAG_E2E=1 to run the index/query tests")
class TestEndToEnd(unittest.TestCase):
    def test_metadata_matches_filename(self):
        # Catches the whole class of "chunk filed under the wrong course" bugs.
        collection = rag._open_collection()
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        self.assertTrue(metadatas)
        for meta in metadatas:
            self.assertTrue(meta["source"].upper().startswith(meta["course_code"]), meta)

    def test_known_course(self):
        result = rag.rag_course_query("COMP4431")
        self.assertTrue(result["answer"])
        self.assertEqual(result["source_files"], ["COMP4431.pdf"])

    def test_question_with_course_code(self):
        result = rag.rag_course_query("What is the assessment scheme of COMP4432?")
        self.assertTrue(result["answer"])
        self.assertIn("COMP4432.pdf", result["source_files"])

    def test_unknown_course(self):
        self.assertEqual(rag.rag_course_query("COMP9999"), {"answer": "", "source_files": []})

    def test_rebuild_is_skipped_when_unchanged(self):
        count = rag.build_index()
        self.assertGreater(count, 0)
        self.assertGreaterEqual(rag.build_index(), count)  # second call reuses the manifest


if __name__ == "__main__":
    unittest.main(verbosity=2)
