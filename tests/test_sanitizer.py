"""sanitizer.py 유닛 테스트."""

import pytest

from harness.sanitizer import DeterministicSanitizer, LocalChunkDB, SanitizeError


class TestLocalChunkDB:
    """LocalChunkDB 테스트."""

    def test_add_and_exact_match(self, chunk_db: LocalChunkDB):
        assert chunk_db.exact_match("서울의 인구는 약 950만 명이다") is True

    def test_exact_match_not_found(self, chunk_db: LocalChunkDB):
        assert chunk_db.exact_match("도쿄의 인구는 1400만 명이다") is False

    def test_fuzzy_match(self, chunk_db: LocalChunkDB):
        # 약간 다른 텍스트도 fuzzy match 가능
        assert chunk_db.fuzzy_match("서울의 인구는 약 950만명이다", threshold=0.8) is True

    def test_fuzzy_match_not_found(self, chunk_db: LocalChunkDB):
        assert chunk_db.fuzzy_match("완전히 다른 문장", threshold=0.85) is False

    def test_url_exists(self, chunk_db: LocalChunkDB):
        assert chunk_db.url_exists("https://kostat.go.kr/report/2024") is True
        assert chunk_db.url_exists("https://fake.com") is False

    def test_number_in_context(self, chunk_db: LocalChunkDB):
        assert chunk_db.number_in_context("950") is True
        assert chunk_db.number_in_context("999") is False

    def test_save_and_reload(self, tmp_dir):
        db_path = tmp_dir / "test_db.json"
        db = LocalChunkDB(db_path=db_path)
        db.add_chunk("테스트 청크", source="test")
        db.add_url("https://test.com")
        db.add_number("42")
        db.save()

        # 새로 로드
        db2 = LocalChunkDB(db_path=db_path)
        assert db2.exact_match("테스트 청크") is True
        assert db2.url_exists("https://test.com") is True
        assert db2.number_in_context("42") is True

    def test_empty_db(self):
        db = LocalChunkDB()
        assert db.exact_match("anything") is False
        assert db.url_exists("https://any.com") is False
        assert db.number_in_context("1") is False


class TestDeterministicSanitizer:
    """DeterministicSanitizer 테스트."""

    def test_valid_citation_no_error(self, sanitizer: DeterministicSanitizer):
        draft = "서울의 인구는 약 950만 명이다. [출처: 서울의 인구는 약 950만 명이다.]"
        errors = sanitizer.validate(draft)
        citation_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(citation_errors) == 0

    def test_fake_citation_detected(self, sanitizer: DeterministicSanitizer):
        draft = "화성의 인구는 100만이다. [출처: 화성 인구 통계 2024]"
        errors = sanitizer.validate(draft)
        citation_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(citation_errors) >= 1

    def test_invalid_url_detected(self, sanitizer: DeterministicSanitizer):
        draft = "참고: https://fake-source.com/data"
        errors = sanitizer.validate(draft)
        url_errors = [e for e in errors if e.error_type == "URL_INVALID"]
        assert len(url_errors) >= 1

    def test_valid_url_no_error(self, sanitizer: DeterministicSanitizer):
        draft = "참고: https://kostat.go.kr/report/2024"
        errors = sanitizer.validate(draft)
        url_errors = [e for e in errors if e.error_type == "URL_INVALID"]
        assert len(url_errors) == 0

    def test_unverified_number_detected(self, sanitizer: DeterministicSanitizer):
        draft = "인플레이션율이 99.5%를 기록했다."
        errors = sanitizer.validate(draft)
        num_errors = [e for e in errors if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(num_errors) >= 1

    def test_verified_number_no_error(self, sanitizer: DeterministicSanitizer):
        draft = "약 950%의 증가율"
        errors = sanitizer.validate(draft)
        num_errors = [e for e in errors if e.error_type == "NUMBER_UNVERIFIED"]
        assert len(num_errors) == 0

    def test_no_citations_no_errors(self, sanitizer: DeterministicSanitizer):
        draft = "이것은 인용이 없는 일반 텍스트입니다."
        errors = sanitizer.validate(draft)
        citation_errors = [e for e in errors if e.error_type == "FAKE_CITATION"]
        assert len(citation_errors) == 0

    def test_validate_as_strings(self, sanitizer: DeterministicSanitizer):
        draft = "[출처: 존재하지 않는 출처] https://fake.com 99.9%"
        results = sanitizer.validate_as_strings(draft)
        assert isinstance(results, list)
        assert all(isinstance(s, str) for s in results)
        assert len(results) >= 1
