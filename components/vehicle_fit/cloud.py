"""Decode unmodified KITTI XYZI and PCD clouds, preserving point correspondence."""
import struct
import numpy as np


def _lzf(data, expected):
    output = bytearray()
    offset = 0
    while offset < len(data):
        control = data[offset]
        offset += 1
        if control < 32:
            length = control + 1
            output.extend(data[offset:offset + length])
            offset += length
        else:
            length = control >> 5
            reference = len(output) - ((control & 31) << 8) - 1
            if length == 7:
                length += data[offset]
                offset += 1
            reference -= data[offset]
            offset += 1
            for index in range(length + 2):
                if reference + index < 0 or reference + index >= len(output):
                    raise ValueError("Invalid LZF back-reference")
                output.append(output[reference + index])
        if len(output) > expected:
            raise ValueError("Invalid compressed PCD size")
    if len(output) != expected:
        raise ValueError("Truncated compressed PCD")
    return bytes(output)


def decode_cloud(data: bytes, cloud_format: str):
    if cloud_format == "bin":
        if not data or len(data) % 16:
            raise ValueError("KITTI cloud must contain float32 XYZI records")
        cloud = np.frombuffer(data, dtype="<f4").reshape(-1, 4).copy()
    elif cloud_format == "pcd":
        lines = []
        cursor = 0
        while cursor < len(data):
            end = data.find(b"\n", cursor)
            if end < 0:
                raise ValueError("Incomplete PCD header")
            line = data[cursor:end].decode("ascii").strip()
            cursor = end + 1
            if line and not line.startswith("#"):
                lines.append(line)
            if line.upper().startswith("DATA "):
                break
        header = {line.split()[0].upper(): line.split()[1:] for line in lines}
        fields = header.get("FIELDS", header.get("FIELD", []))
        if not all(name in fields for name in ("x", "y", "z")):
            raise ValueError("PCD must contain x, y and z")
        sizes = [int(n) for n in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(n) for n in header.get("COUNT", ["1"] * len(fields))]
        points = int(header.get("POINTS", [str(int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0]))])[0])
        if not 0 < points <= 2_000_000:
            raise ValueError("Invalid PCD point count")
        if not len(fields) == len(sizes) == len(types) == len(counts):
            raise ValueError("Inconsistent PCD fields")
        descriptors = []
        for field, size, kind, count in zip(fields, sizes, types, counts):
            if kind not in ("F", "I", "U") or size not in (1, 2, 4, 8) or count < 1:
                raise ValueError("Unsupported PCD field")
            dtype = "<" + {"F": "f", "I": "i", "U": "u"}[kind] + str(size)
            descriptors.append((field, dtype) if count == 1 else (field, dtype, (count,)))
        mode = header["DATA"][0].lower()
        payload = data[cursor:]
        if mode == "ascii":
            array = np.loadtxt(payload.splitlines(), dtype=np.float32).reshape(points, sum(counts))
            offsets = np.cumsum([0] + counts[:-1])
            values = {field: array[:, offset] for field, offset in zip(fields, offsets)}
        elif mode == "binary":
            dtype = np.dtype(descriptors)
            if len(payload) != points * dtype.itemsize:
                raise ValueError("PCD binary payload size mismatch")
            values = np.frombuffer(payload, dtype=dtype, count=points)
        elif mode == "binary_compressed":
            if len(payload) < 8:
                raise ValueError("Truncated compressed PCD")
            compressed, uncompressed = struct.unpack("<II", payload[:8])
            expected = points * sum(size * count for size, count in zip(sizes, counts))
            if uncompressed != expected or compressed != len(payload) - 8:
                raise ValueError("PCD compressed payload size mismatch")
            unpacked = _lzf(payload[8:], expected)
            values = {}
            offset = 0
            for descriptor, size, count in zip(descriptors, sizes, counts):
                field, dtype = descriptor[:2]
                length = points * size * count
                values[field] = np.frombuffer(unpacked[offset:offset + length], dtype=dtype).reshape(points, count)[:, 0]
                offset += length
        else:
            raise ValueError("Unsupported PCD DATA mode")
        cloud = np.zeros((points, 4), dtype=np.float32)
        for column, field in enumerate(("x", "y", "z", "intensity")):
            if field in fields:
                value = values[field]
                cloud[:, column] = value[:, 0] if value.ndim == 2 else value
    else:
        raise ValueError("cloud_format must be bin or pcd")
    # Match CVAT's removal of non-finite XYZ without treating intensity as geometry.
    finite = np.isfinite(cloud[:, :3]).all(axis=1)
    original_indices = np.flatnonzero(finite)
    cloud = cloud[finite]
    cloud[:, 3] = np.nan_to_num(cloud[:, 3])
    if not len(cloud):
        raise ValueError("No finite cloud points")
    return cloud, original_indices
