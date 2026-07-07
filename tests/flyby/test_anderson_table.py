from oph_fpe.flyby import anderson_mm_s


def test_near_anderson_public_value() -> None:
    assert abs(anderson_mm_s(6.851, -20.76, -71.96) - 13.278) < 0.01


def test_galileo_i_anderson_public_value() -> None:
    assert abs(anderson_mm_s(8.949, -12.52, -34.15) - 4.122) < 0.01
