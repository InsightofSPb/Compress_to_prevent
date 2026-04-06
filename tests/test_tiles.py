from compression.tiles import tile_scores


def test_tile_indexing_determinism() -> None:
    width, height = 8, 8
    residual = (width, height, bytes([0] * (width * height * 3)))
    first = tile_scores(residual, tile_size=4)
    second = tile_scores(residual, tile_size=4)
    assert first == second
    assert [item[:2] for item in first] == [(0, 0), (1, 0), (0, 1), (1, 1)]
