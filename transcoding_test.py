"""Tests for transcoding command generation."""

import pytest

from transcoding import HwAccel, _build_audio_args, _build_video_args, build_hls_ffmpeg_cmd


class FakeMediaInfo:
    """Fake media info for testing."""

    def __init__(
        self,
        video_codec: str = "h264",
        audio_codec: str = "aac",
        pix_fmt: str = "yuv420p",
        audio_channels: int = 2,
        audio_sample_rate: int = 48000,
        height: int = 1080,
    ):
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.pix_fmt = pix_fmt
        self.audio_channels = audio_channels
        self.audio_sample_rate = audio_sample_rate
        self.height = height


class TestBuildVideoArgs:
    """Tests for _build_video_args."""

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
    @pytest.mark.parametrize("deinterlace", [True, False])
    @pytest.mark.parametrize("max_resolution", ["1080p", "720p", "4k"])
    def test_all_hw_combinations(self, hw: HwAccel, deinterlace: bool, max_resolution: str):
        """Test all hardware/deinterlace/resolution combinations produce valid args."""
        pre, post = _build_video_args(
            copy_video=False,
            hw=hw,
            deinterlace=deinterlace,
            use_hw_pipeline=(hw != "software"),
            max_resolution=max_resolution,
            quality="high",
        )

        # Pre args: hw accelerators have hwaccel, software has none
        if hw == "nvidia":
            assert pre == [] or "-hwaccel" in pre
        elif hw == "intel":
            assert "-hwaccel" in pre
            assert "qsv" in pre
        elif hw == "vaapi":
            assert "-hwaccel" in pre
            assert "vaapi" in pre
        else:
            assert pre == []

        # Post args: always have -vf, -c:v, encoder, -g
        assert "-vf" in post
        assert "-c:v" in post
        assert "-g" in post
        assert "60" in post

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
    def test_copy_video(self, hw: HwAccel):
        """Test copy_video returns minimal args."""
        pre, post = _build_video_args(
            copy_video=True,
            hw=hw,
            deinterlace=False,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        assert post == ["-c:v", "copy"]

    def test_nvidia_hw_pipeline_filters(self):
        """Test NVIDIA with hw pipeline uses CUDA filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvidia",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        assert "-hwaccel" in pre
        vf = post[post.index("-vf") + 1]
        assert "yadif_cuda" in vf
        assert "scale_cuda" in vf

    def test_nvidia_sw_fallback_filters(self):
        """Test NVIDIA without hw pipeline uses software filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="nvidia",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        vf = post[post.index("-vf") + 1]
        assert "yadif=1" in vf
        assert "cuda" not in vf

    def test_vaapi_filters(self):
        """Test VAAPI uses VAAPI filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="vaapi",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        vf = post[post.index("-vf") + 1]
        assert "deinterlace_vaapi" in vf
        assert "scale_vaapi" in vf

    def test_intel_filters(self):
        """Test Intel uses QSV filters."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="intel",
            deinterlace=True,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality="high",
        )
        vf = post[post.index("-vf") + 1]
        assert "vpp_qsv" in vf
        assert "scale_qsv" in vf

    def test_software_filters(self):
        """Test software uses yadif and scale."""
        pre, post = _build_video_args(
            copy_video=False,
            hw="software",
            deinterlace=True,
            use_hw_pipeline=False,
            max_resolution="1080p",
            quality="high",
        )
        assert pre == []
        vf = post[post.index("-vf") + 1]
        assert "yadif=1" in vf

    @pytest.mark.parametrize(
        "quality,expected_qp", [("high", "20"), ("medium", "28"), ("low", "35")]
    )
    def test_quality_presets(self, quality: str, expected_qp: str):
        """Test quality presets map to correct QP values."""
        _, post = _build_video_args(
            copy_video=False,
            hw="vaapi",
            deinterlace=False,
            use_hw_pipeline=True,
            max_resolution="1080p",
            quality=quality,
        )
        assert expected_qp in post

    def test_invalid_hw_raises(self):
        """Test invalid hardware raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized hardware"):
            _build_video_args(
                copy_video=False,
                hw="invalid",  # type: ignore
                deinterlace=False,
                use_hw_pipeline=False,
                max_resolution="1080p",
                quality="high",
            )


class TestBuildAudioArgs:
    """Tests for _build_audio_args."""

    def test_copy_audio(self):
        """Test copy_audio returns copy args."""
        args = _build_audio_args(copy_audio=True, audio_sample_rate=48000)
        assert args == ["-c:a", "copy"]

    @pytest.mark.parametrize(
        "sample_rate,expected",
        [
            (44100, "44100"),
            (48000, "48000"),
            (96000, "48000"),  # non-standard falls back to 48000
            (0, "48000"),
        ],
    )
    def test_sample_rates(self, sample_rate: int, expected: str):
        """Test sample rate handling."""
        args = _build_audio_args(copy_audio=False, audio_sample_rate=sample_rate)
        assert "-ar" in args
        assert expected in args


class TestBuildHlsFfmpegCmd:
    """Tests for build_hls_ffmpeg_cmd."""

    @pytest.mark.parametrize("hw", ["nvidia", "intel", "vaapi", "software"])
    @pytest.mark.parametrize("is_vod", [True, False])
    def test_command_structure(self, hw: HwAccel, is_vod: bool):
        """Test command has correct structure for all hw/vod combinations."""
        cmd = build_hls_ffmpeg_cmd(
            "http://test/stream",
            hw,
            "/tmp/output",
            is_vod=is_vod,
        )

        # Basic structure
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert "-map" in cmd
        assert "-c:v" in cmd
        assert "-c:a" in cmd
        assert "-f" in cmd
        assert "hls" in cmd

        # hwaccel before -i for hw encoders
        i_idx = cmd.index("-i")
        if "-hwaccel" in cmd:
            hwaccel_idx = cmd.index("-hwaccel")
            assert hwaccel_idx < i_idx, "hwaccel must come before -i"

        # -vf after -i
        if "-vf" in cmd:
            vf_idx = cmd.index("-vf")
            assert vf_idx > i_idx, "-vf must come after -i"

    def test_vod_hls_flags(self):
        """Test VOD has correct HLS flags."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=True)
        assert "-hls_playlist_type" in cmd
        assert "event" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == "0"

    def test_live_hls_flags(self):
        """Test live has correct HLS flags."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", is_vod=False)
        assert "delete_segments" in cmd
        assert "-hls_list_size" in cmd
        assert cmd[cmd.index("-hls_list_size") + 1] == "10"

    def test_copy_video_with_compatible_media(self):
        """Test copy_video is used for compatible VOD media."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution="1080p",
        )
        # Should copy video, no hwaccel needed
        assert "-c:v" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert "-hwaccel" not in cmd

    def test_no_copy_for_10bit(self):
        """Test 10-bit content is transcoded, not copied."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p10le", height=1080)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"
        assert "-hwaccel" in cmd

    def test_no_copy_when_scaling_needed(self):
        """Test scaling requirement prevents copy."""
        media = FakeMediaInfo(video_codec="h264", pix_fmt="yuv420p", height=2160)
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution="1080p",
        )
        assert cmd[cmd.index("-c:v") + 1] != "copy"

    def test_user_agent(self):
        """Test user agent is included when provided."""
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "software",
            "/tmp",
            user_agent="TestAgent/1.0",
        )
        assert "-user_agent" in cmd
        assert "TestAgent/1.0" in cmd

    def test_probe_args_without_media_info(self):
        """Test probe args are added when no media_info."""
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", media_info=None)
        assert "-probesize" in cmd
        assert "-analyzeduration" in cmd

    def test_no_probe_args_with_media_info(self):
        """Test probe args are skipped when media_info provided."""
        media = FakeMediaInfo()
        cmd = build_hls_ffmpeg_cmd("http://test", "software", "/tmp", media_info=media)  # type: ignore
        assert "-probesize" not in cmd


class TestAspectRatioHandling:
    """Tests for various aspect ratio content."""

    @pytest.mark.parametrize(
        "input_height,max_res,should_scale",
        [
            (1080, "1080p", False),  # exact match - no min(ih,X) needed
            (1080, "720p", True),  # needs scale down
            (720, "1080p", False),  # smaller than max, but min(ih,1080) still present
            (2160, "1080p", True),  # 4K to 1080p
            (1600, "1080p", True),  # ultrawide 4K to 1080p
            (1600, "4k", False),  # ultrawide 4K, no scale needed
        ],
    )
    def test_scaling_decisions(self, input_height: int, max_res: str, should_scale: bool):
        """Test correct scaling decisions for various input heights."""
        media = FakeMediaInfo(height=input_height, pix_fmt="yuv420p10le")
        cmd = build_hls_ffmpeg_cmd(
            "http://test",
            "vaapi",
            "/tmp",
            is_vod=True,
            media_info=media,  # type: ignore
            max_resolution=max_res,
        )
        vf = cmd[cmd.index("-vf") + 1]
        # Check filter contains height constraint
        from transcoding import _MAX_RES_HEIGHT

        max_h = _MAX_RES_HEIGHT.get(max_res, 9999)
        height_expr = f"min(ih,{max_h})"
        assert height_expr in vf, f"Expected {height_expr} in {vf}"


if __name__ == "__main__":
    from testing import run_tests

    run_tests(__file__)
