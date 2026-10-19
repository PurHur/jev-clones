"""Versioned, bounded local image admission. Never opens a URL or unregistered path."""
import hashlib
import io
from pathlib import Path
import re
import stat
import warnings
from PIL import Image

SCHEMA = 'lemonjev-local-images-v1'
POLICY = 'registered-local-v1'
MAX_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_IMAGES = 8
IMAGE_SIZE = {'shortest_edge': 65536, 'longest_edge': 262144}


def asset_record(record):
    if not isinstance(record, dict) or set(record) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Image assets require exactly path, sha256, bytes')
    path = record['path']
    if not isinstance(path, str) or not Path(path).is_absolute() or str(Path(path)) != path or '..' in Path(path).parts:
        raise ValueError('Image path must be a canonical absolute local path')
    if not isinstance(record['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
        raise ValueError('Invalid image SHA256')
    if type(record['bytes']) is not int or not 0 < record['bytes'] <= MAX_BYTES:
        raise ValueError('Image byte bound exceeded')
    return dict(record)


class LocalImageRegistry:
    def __init__(self, manifest):
        if not isinstance(manifest, dict) or set(manifest) != {'schema', 'images'} or manifest['schema'] != SCHEMA:
            raise ValueError('Missing versioned local_images manifest')
        if not isinstance(manifest['images'], list):
            raise ValueError('Image registry must be a list')
        self.assets = {}
        for value in manifest['images']:
            record = asset_record(value)
            if record['path'] in self.assets:
                raise ValueError('Duplicate registered image path')
            self.assets[record['path']] = record

    def records(self, row):
        paths = row.get('images', [])
        records = row.get('source', {}).get('image_assets', [])
        if not isinstance(paths, list) or len(paths) > MAX_IMAGES or not isinstance(records, list) or len(paths) != len(records):
            raise ValueError('Images/image_assets must correspond one-to-one within image limit')
        result = []
        for path, value in zip(paths, records, strict=True):
            record = asset_record(value)
            if not isinstance(path, str) or path != record['path'] or self.assets.get(path) != record:
                raise ValueError('Unregistered or mismatched row image asset')
            result.append(record)
        return result

    def decode(self, record):
        record = asset_record(record)
        if self.assets.get(record['path']) != record:
            raise ValueError('Unregistered image')
        path = Path(record['path'])
        # No symlinks, special devices or unbounded streams. Decode these verified
        # bytes, not a second path open, so a replacement cannot evade the hash.
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError('Image symlinks are not allowed')
        import os
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != record['bytes']:
                raise ValueError('Image size/type mismatch')
            data = stream.read(MAX_BYTES + 1)
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Image size/hash mismatch')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > MAX_PIXELS or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('Unsupported or oversized image')
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    return image.convert('RGB')
        except (OSError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
            raise ValueError('Corrupt or oversized registered image') from exc

    def decode_row(self, row):
        return [self.decode(record) for record in self.records(row)]


def configure_local_images(processor, registry):
    if not hasattr(processor, 'image_processor'):
        raise ValueError('Registered local images require a multimodal processor')
    processor.image_processor.size = dict(IMAGE_SIZE)
    processor.local_image_registry = registry
    return processor
