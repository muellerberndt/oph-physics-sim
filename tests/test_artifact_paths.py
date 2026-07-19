from pathlib import Path

from oph_fpe.evidence.artifact_paths import companion_input_packet_path


def test_companion_input_packet_paths_preserve_canonical_name_and_avoid_collisions():
    directory = Path("run")

    canonical = companion_input_packet_path(
        directory / "fair_block_certificate.json",
        canonical_certificate_filename="fair_block_certificate.json",
        canonical_input_filename="fair_block_input_packet.json",
    )
    first = companion_input_packet_path(
        directory / "first.json",
        canonical_certificate_filename="fair_block_certificate.json",
        canonical_input_filename="fair_block_input_packet.json",
    )
    second = companion_input_packet_path(
        directory / "second.json",
        canonical_certificate_filename="fair_block_certificate.json",
        canonical_input_filename="fair_block_input_packet.json",
    )

    assert canonical == directory / "fair_block_input_packet.json"
    assert first == directory / "first_input_packet.json"
    assert second == directory / "second_input_packet.json"
    assert len({canonical, first, second}) == 3
