### Label File Format

The label file is a list composed of multiple dictionaries, with each dictionary representing one sequence.

Each sequence consists of three parts: `"labels"`, `"boxes"`, and `"seq_len"`\
Example:

```json
{"labels": [1], "boxes": [[155.0, 27]], "seq_len": 200}
```

- `"labels"`: The object label(s). Set to `0` if the sequence is a negative sample.

- `"boxes"`: The ground truth bounding box positions in the format:\
  `[[center_x, obj_length], [center_x, obj_length], ...]`

  - `center_x`: The center position of the object within the sequence
  - `obj_length`: The length of the object\
    If the sequence is a negative sample, all values should be set to `0`.

- `"seq_len"`: The total length of the sequence.
