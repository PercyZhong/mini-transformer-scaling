from mini_transformer.data import nested_window_subsets, split_text


def test_sequential_split_ranges_do_not_overlap() -> None:
    splits = split_text("abcdefghijklmnopqrstuvwxyz" * 10)
    assert splits.train_range[1] == splits.validation_range[0]
    assert splits.validation_range[1] == splits.test_range[0]
    assert splits.train + splits.validation + splits.test == "abcdefghijklmnopqrstuvwxyz" * 10


def test_window_subsets_are_nested_and_reproducible() -> None:
    first = nested_window_subsets(1000, 32, seed=42)
    second = nested_window_subsets(1000, 32, seed=42)
    assert first == second
    assert set(first[0.1]) < set(first[0.3]) < set(first[1.0])
    assert len(first[0.1]) == int((1000 - 32) * 0.1)
    assert len(first[0.3]) == int((1000 - 32) * 0.3)
    assert len(first[1.0]) == 1000 - 32
