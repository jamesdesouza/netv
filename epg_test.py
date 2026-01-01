"""Tests for epg.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import epg


class TestParseEpgTime:
    def test_parse_basic_time(self):
        result = epg.parse_epg_time("20241130120000")
        assert result.year == 2024
        assert result.month == 11
        assert result.day == 30
        assert result.hour == 12
        assert result.minute == 0
        assert result.second == 0

    def test_parse_time_with_positive_offset(self):
        result = epg.parse_epg_time("20241130120000 +0530")
        assert result.hour == 12
        assert result.tzinfo is not None
        assert result.utcoffset() == timedelta(hours=5, minutes=30)

    def test_parse_time_with_negative_offset(self):
        result = epg.parse_epg_time("20241130120000-0500")
        assert result.utcoffset() == timedelta(hours=-5)

    def test_parse_time_with_zero_offset(self):
        result = epg.parse_epg_time("20241130120000 +0000")
        assert result.utcoffset() == timedelta(0)

    def test_parse_invalid_returns_now(self):
        result = epg.parse_epg_time("invalid")
        assert (datetime.now(UTC) - result).total_seconds() < 2


class TestProgram:
    def test_program_dataclass(self):
        prog = epg.Program(
            channel_id="ch1",
            title="Test Show",
            start=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            stop=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            desc="Description",
        )
        assert prog.channel_id == "ch1"
        assert prog.title == "Test Show"
        assert prog.source_id == ""


class TestEPGData:
    def test_epg_data_defaults(self):
        epg_data = epg.EPGData()
        assert epg_data.channels == {}
        assert epg_data.icons == {}
        assert epg_data.programs == {}


class TestGetProgramsInRange:
    def test_programs_in_range(self):
        epg_data = epg.EPGData()
        epg_data.programs["ch1"] = [
            epg.Program(
                "ch1",
                "Show 1",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
            ),
            epg.Program(
                "ch1",
                "Show 2",
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            ),
            epg.Program(
                "ch1",
                "Show 3",
                datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            ),
        ]

        # Get programs from 10:30 to 11:30
        result = epg.get_programs_in_range(
            epg_data,
            "ch1",
            datetime(2024, 1, 1, 10, 30, tzinfo=UTC),
            datetime(2024, 1, 1, 11, 30, tzinfo=UTC),
        )
        assert len(result) == 2
        assert result[0].title == "Show 1"
        assert result[1].title == "Show 2"

    def test_programs_empty_channel(self):
        epg_data = epg.EPGData()
        result = epg.get_programs_in_range(
            epg_data,
            "nonexistent",
            datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        )
        assert result == []

    def test_programs_prefer_source(self):
        epg_data = epg.EPGData()
        epg_data.programs["ch1"] = [
            epg.Program(
                "ch1",
                "Show A",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                source_id="source1",
            ),
            epg.Program(
                "ch1",
                "Show B",
                datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 11, 0, tzinfo=UTC),
                source_id="source2",
            ),
        ]

        result = epg.get_programs_in_range(
            epg_data,
            "ch1",
            datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            preferred_source_id="source2",
        )
        assert len(result) == 1
        assert result[0].title == "Show B"


class TestSanitizeEpgXml:
    def test_sanitize_extracts_channels_and_programmes(self):
        xml = """
        <tv>
        <channel id="ch1"><display-name>Channel 1</display-name></channel>
        <corrupted>
        <programme start="20240101120000 +0000" stop="20240101130000 +0000" channel="ch1">
            <title>Test</title>
        </programme>
        """
        result = epg._sanitize_epg_xml(xml)
        assert "<channel" in result
        assert "<programme" in result
        assert '<?xml version="1.0"?>' in result


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
