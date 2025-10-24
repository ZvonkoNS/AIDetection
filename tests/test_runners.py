from aidetect.runner.batch import find_images


def test_find_images_non_recursive(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    file_top = image_root / "top.jpg"
    file_top.write_bytes(b"test")

    nested = image_root / "nested"
    nested.mkdir()
    file_nested = nested / "nested.jpg"
    file_nested.write_bytes(b"test")

    results = find_images(str(image_root), recursive=False)
    assert str(file_top.resolve()) in results
    assert str(file_nested.resolve()) not in results


def test_find_images_recursive(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "top.JPG").write_text("test")
    nested = image_root / "nested"
    nested.mkdir()
    nested_file = nested / "nested.PNG"
    nested_file.write_text("test")

    results = find_images(str(image_root), recursive=True)
    assert str(nested_file.resolve()) in results
