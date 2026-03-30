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

    def test_unverified_date_detected(self):
        """KB에 없는 날짜가 DATE_UNVERIFIED로 탐지되는지 확인."""
        db = LocalChunkDB()
        db.add_chunk("SN8은 2020년 12월 9일 첫 비행을 했다.", source="spacex")
        sanitizer = DeterministicSanitizer(chunk_db=db)

        # KB에 없는 날짜 → DATE_UNVERIFIED
        draft = "SN8은 2023년 4월 20일 첫 비행 테스트를 진행했습니다."
        errors = sanitizer.validate(draft)
        date_errors = [e for e in errors if e.error_type == "DATE_UNVERIFIED"]
        assert len(date_errors) == 1
        assert "2023년 4월 20일" in date_errors[0].description

    def test_verified_date_no_error(self):
        """KB에 존재하는 날짜는 오류를 발생시키지 않는지 확인."""
        db = LocalChunkDB()
        db.add_chunk("SN8은 2020년 12월 9일 첫 비행을 했다.", source="spacex")
        sanitizer = DeterministicSanitizer(chunk_db=db)

        draft = "SN8은 2020년 12월 9일 첫 비행을 했다."
        errors = sanitizer.validate(draft)
        date_errors = [e for e in errors if e.error_type == "DATE_UNVERIFIED"]
        assert len(date_errors) == 0

    def test_date_check_skipped_when_no_kb(self):
        """KB가 비어있으면 날짜 검증을 건너뛴다."""
        db = LocalChunkDB()  # 빈 DB
        sanitizer = DeterministicSanitizer(chunk_db=db)

        draft = "SN8은 2023년 4월 20일 첫 비행을 했다."
        errors = sanitizer.validate(draft)
        date_errors = [e for e in errors if e.error_type == "DATE_UNVERIFIED"]
        assert len(date_errors) == 0  # 빈 KB에서는 건너뜀

    def test_date_multiple_formats(self):
        """다양한 날짜 표기를 정확히 매칭하는지 확인."""
        db = LocalChunkDB()
        db.add_chunk("2023-04-20에 IFT-1 발사", source="spacex")
        sanitizer = DeterministicSanitizer(chunk_db=db)

        # "2023년 4월 20일" → "2023-04-20" 변환 후 매칭
        draft = "IFT-1은 2023년 4월 20일 발사되었다."
        errors = sanitizer.validate(draft)
        date_errors = [e for e in errors if e.error_type == "DATE_UNVERIFIED"]
        assert len(date_errors) == 0
