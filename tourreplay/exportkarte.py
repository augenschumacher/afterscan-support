"""Build TourReplay’s read-only ODbL map from public Protomaps basemap extracts.

Python 3.10+, pmtiles CLI 1.31.2 on PATH. Install Python packages with:
  python -m pip install pmtiles==3.7.0 mapbox-vector-tile==2.2.0 shapely==2.2.0 pyshp==3.1.8
Run: python exportkarte.py --cache /tmp/tourreplay-map ExportMap.sqlite
No tour, device data or user coordinates are required. See kartendaten.html.
"""
import argparse
import functools
import gzip
import hashlib
import math
from pathlib import Path
import sqlite3
import struct
import subprocess
import tempfile
import urllib.request

import shapefile
from mapbox_vector_tile.decoder import TileData
from pmtiles.reader import MmapSource, Reader, all_tiles
from shapely import clip_by_rect, constrained_delaunay_triangles, make_valid
from shapely.geometry import shape

SOURCE = 'https://build.protomaps.com/20261114.pmtiles'
REGIONS = (
    ('world', 5, (-179.99999, -85, 179.99999, 85)),
    ('dach', 10, (5.85, 45.75, 17.18, 55.1)),
    ('stuttgart', 13, (8.96, 48.62, 9.40, 48.94)),
)
LICENSE = 'https://opendatacommons.org/licenses/odbl/1-0/'
ORIGIN = '© OpenStreetMap contributors; https://www.openstreetmap.org/copyright; Protomaps basemaps; Natural Earth (public domain)'
NATURAL_EARTH = (
    ('roads', '10m_cultural/ne_10m_roads.zip', '880b235383a1e1ccf33f4ea6f82b6755a2ffbce07e3974e412b43e88fe954ae9'),
    ('rivers', '10m_physical/ne_10m_rivers_lake_centerlines.zip', 'ded71b01870855ccfe19b51f2ec14c9bb48fae23c0e9f3c11974d426433b5c38'),
)


def fetch(path, zoom, box, source):
    if path.is_file():
        return
    temporary = path.with_suffix('.partial.pmtiles')
    temporary.unlink(missing_ok=True)
    subprocess.run(['pmtiles', 'extract', source, str(temporary),
                    '--bbox=' + ','.join(map(str, box)), f'--minzoom={zoom}',
                    f'--maxzoom={zoom}', '--download-threads=2'], check=True)
    temporary.replace(path)


def feature_kind(layer, props, polygon, level):
    kind = props.get('kind')
    detail = props.get('kind_detail')
    if layer == 'roads' and not polygon:
        if level == 0:  # Use Natural Earth's more extensive global roads layer below.
            return None
        if kind == 'highway' or kind == 'major_road' and detail in {'primary', 'trunk', 'primary_link', 'trunk_link'}:
            return 1
        if kind in {'major_road', 'minor_road'}:
            return 2
        if level == 2 and kind == 'path' and detail in {'cycleway', 'path', 'track', 'pedestrian'}:
            return 2
    if layer == 'water' and kind not in {'ocean', 'sea', 'swimming_pool', 'fountain'}:
        return 4 if polygon else 3
    if layer == 'earth' and kind == 'earth' and polygon:
        return 5
    if layer == 'landuse' and level == 2 and polygon and kind in {'wood', 'forest', 'park', 'nature_reserve', 'garden'}:
        return 6
    return None


def global_lines(db, cache):
    for name, remote, expected in NATURAL_EARTH:
        archive = cache / (name + '.zip')
        url = 'https://naturalearth.s3.amazonaws.com/' + remote
        if not archive.is_file():
            urllib.request.urlretrieve(url, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Natural Earth source fingerprint changed for {name}')
        db.executemany('INSERT INTO metadata VALUES (?,?)', [('naturalearth:' + name, url), (name + ':sha256', expected)])
        with shapefile.Reader(str(archive)) as source:
            for entry in source.iterShapeRecords():
                props = entry.record
                if name == 'roads':
                    road_type = props['type']
                    if road_type.startswith('Ferry'):
                        continue
                    kind = 1 if road_type in {'Major Highway', 'Beltway', 'Bypass'} else 2
                else:
                    if props['featurecla'] == 'Lake Centerline':
                        continue
                    kind = 3
                geom = shape(entry.shape.__geo_interface__).simplify(.0015, preserve_topology=False)
                for coords in pieces(geom, polygon=False):
                    # Short bounding boxes keep regional lookups inexpensive on long rivers.
                    for start in range(0, len(coords) - 1, 63):
                        points = list(coords[start:start + 64])
                        if len(points) < 2:
                            continue
                        xs, ys = zip(*points)
                        binary = b''.join(struct.pack('<ff', x, y) for x, y in points)
                        insert_feature(db, (kind, props['name'] or None, binary, 0), (min(xs), max(xs), min(ys), max(ys)))


def insert_feature(db, record, bounds):
    rowid = db.execute('INSERT INTO feature(kind,name,geom,level) VALUES (?,?,?,?)', record).lastrowid
    db.execute('INSERT INTO bounds VALUES (?,?,?,?,?)', (rowid, *bounds))


def pieces(geom, polygon):
    if geom.is_empty:
        return
    if polygon and geom.geom_type == 'Polygon':
        if not geom.interiors:
            yield geom.exterior.coords
            return
        # The on-device format uses one exterior ring per surface. Split polygons with
        # holes into adjacent filled faces so islands and park clearings remain visible.
        min_x, min_y, max_x, max_y = geom.bounds
        hole = max(geom.interiors, key=lambda ring: ring.length)
        west, _, east, _ = hole.bounds
        split_x = (west + east) / 2
        if not min_x < split_x < max_x or len(geom.interiors) > 80:
            for face in constrained_delaunay_triangles(geom).geoms:
                yield face.exterior.coords
            return
        for left, right in ((min_x, split_x), (split_x, max_x)):
            yield from pieces(clip_by_rect(geom, left, min_y, right, max_y), polygon)
    elif not polygon and geom.geom_type == 'LineString':
        yield geom.coords
    elif geom.geom_type.startswith('Multi') or geom.geom_type == 'GeometryCollection':
        for part in geom.geoms:
            yield from pieces(part, polygon)


def tile_records(z, x, y, blob, level):
    data = TileData(gzip.decompress(blob))
    for layer in data.tile.layers:
        if layer.name not in {'earth', 'water', 'roads'} and not (level == 2 and layer.name == 'landuse'):
            continue
        extent = layer.extent
        global_extent = extent * 2 ** z

        @functools.lru_cache(maxsize=16384)
        def latitude(tile_y):
            return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y * extent + tile_y) / global_extent))))

        def longitude(tile_x):
            return (x * extent + tile_x) / global_extent * 360 - 180

        for feature in layer.features:
            if feature.type not in (2, 3):
                continue
            tags = iter(feature.tags)
            props = {layer.keys[key]: data.parse_value(layer.values[value]) for key, value in zip(tags, tags)
                     if layer.keys[key] in {'kind', 'kind_detail', 'name', 'name:de'}}
            polygon = feature.type == 3
            kind = feature_kind(layer.name, props, polygon, level)
            if kind is None:
                continue
            decoded = data.parse_geometry(feature.geometry, feature.type, extent, True, None)
            geometry = shape(decoded)
            if geometry.is_empty:
                continue
            geometry = clip_by_rect(geometry, 0., 0., float(extent), float(extent))
            if geometry.is_empty:
                continue
            if polygon and not geometry.is_valid:
                geometry = make_valid(geometry)
            tolerance = (1.0, 2.5, 2.0)[level]
            geometry = geometry.simplify(tolerance, preserve_topology=polygon)
            name = props.get('name:de') or props.get('name')
            for coords in pieces(geometry, polygon):
                vertices = [(longitude(u), latitude(v)) for u, v in coords]
                if len(vertices) < (4 if polygon else 2):
                    continue
                xy = list(zip(*vertices))
                geom = b''.join(struct.pack('<ff', lon, lat) for lon, lat in vertices)
                yield (kind, name, geom, level), (min(xy[0]), max(xy[0]), min(xy[1]), max(xy[1]))


def build(output, cache, source):
    cache.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as working_dir:
        temp = Path(working_dir) / 'ExportMap.sqlite'
        with sqlite3.connect(temp) as db:
            db.executescript('''
                PRAGMA application_id=1414677840;
                PRAGMA user_version=1;
                CREATE TABLE feature(id INTEGER PRIMARY KEY,kind INTEGER NOT NULL,name TEXT,geom BLOB NOT NULL,level INTEGER NOT NULL);
                CREATE VIRTUAL TABLE bounds USING rtree(id,minLon,maxLon,minLat,maxLat);
                CREATE TABLE coverage(level INTEGER PRIMARY KEY,minLon REAL,maxLon REAL,minLat REAL,maxLat REAL);
                CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            ''')
            db.executemany('INSERT INTO metadata VALUES (?,?)', [('license', LICENSE), ('attribution', ORIGIN), ('source', source)])
            for level, (name, zoom, box) in enumerate(REGIONS):
                filename = cache / (name + '.pmtiles')
                fetch(filename, zoom, box, source)
                with filename.open('rb') as archive:
                    digest = hashlib.sha256()
                    for block in iter(lambda: archive.read(1024 * 1024), b''):
                        digest.update(block)
                    archive.seek(0)
                    content = MmapSource(archive)
                    metadata = Reader(content).metadata()
                    for key in ('version', 'planetiler:osm:osmosisreplicationtime'):
                        db.execute('INSERT INTO metadata VALUES (?,?)', (name + ':' + key, metadata[key]))
                    db.execute('INSERT INTO metadata VALUES (?,?)', (name + ':sha256', digest.hexdigest()))
                    west, south, east, north = box
                    if level == 0:
                        west, south, east, north = -180., -85.05112878, 180., 85.05112878
                    db.execute('INSERT INTO coverage VALUES (?,?,?,?,?)', (level, west, east, south, north))
                    for i, ((z, x, y), data) in enumerate(all_tiles(content), 1):
                        if z != zoom:
                            raise ValueError(f'Unexpected zoom {z} in {filename}')
                        for record, bounds in tile_records(z, x, y, data, level):
                            insert_feature(db, record, bounds)
                        if i % 250 == 0:
                            print(name, i, flush=True)
                if level == 0:
                    global_lines(db, cache)
                print(name, db.execute('SELECT kind,count(*) FROM feature WHERE level=? GROUP BY kind', (level,)).fetchall(), flush=True)
                db.commit()
            assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            assert db.execute("SELECT rtreecheck('bounds')").fetchone() == ('ok',)
        temp.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cache', required=True, type=Path, help='Separate cache directory for each source snapshot')
    parser.add_argument('--source', default=SOURCE, help='Public Protomaps v4 planet URL for an updated build')
    args = parser.parse_args()
    build(args.output, args.cache, args.source)
