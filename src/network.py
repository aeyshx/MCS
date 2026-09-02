import os
import warnings
import osmnx as ox
import networkx as nx
import geopandas as gpd
from .paths import ProjectPaths

# Suppress deprecation warning for north/south/east/west kwargs in OSMnx 1.9.x
warnings.filterwarnings(
    "ignore",
    message=".*north.*south.*east.*west.*deprecated.*",
    category=FutureWarning,
)

# Bounding box for Legazpi-Daraga corridor
NORTH, SOUTH = 13.20, 13.09
EAST, WEST = 123.78, 123.68

CACHE_GRAPH_PATH = ProjectPaths.discover().cache / "legazpi_daraga_graph.graphml"

from .demand import ROUTES  # single source of truth for route identifiers


def nearest_graph_node(G, lon, lat):
    """Return the closest graph node without requiring scikit-learn.

    OSMnx delegates geographic nearest-neighbour searches to scikit-learn,
    which is an optional dependency.  The small cached study graph is readily
    handled by a NumPy-free linear fallback when that optional package is not
    installed.
    """
    try:
        return ox.nearest_nodes(G, lon, lat)
    except ImportError:
        return min(
            G.nodes,
            key=lambda node: (G.nodes[node]['x'] - lon) ** 2 + (G.nodes[node]['y'] - lat) ** 2,
        )

def build_road_graph(use_cache=True):
    '''Download (or load cached) and simplify the road network for the corridor.'''
    if use_cache and os.path.exists(CACHE_GRAPH_PATH):
        G = ox.load_graphml(CACHE_GRAPH_PATH)
        return G

    G = ox.graph_from_bbox(
        north=NORTH, south=SOUTH, east=EAST, west=WEST,
        network_type='drive',
        simplify=True,
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    if use_cache:
        os.makedirs(os.path.dirname(CACHE_GRAPH_PATH), exist_ok=True)
        ox.save_graphml(G, CACHE_GRAPH_PATH)

    return G

def load_routes(geojson_path=None):
    '''Load route geometries as GeoDataFrame.'''
    return gpd.read_file(geojson_path or ProjectPaths.discover().route_geometries, engine="pyogrio")

def route_edges(G, route_geom):
    '''Map a route geometry to a sequence of edges in the road graph.'''
    # Find nearest graph edges for each vertex of the route polyline
    coords = list(route_geom.coords)
    edges = []
    for u_coord, v_coord in zip(coords[:-1], coords[1:]):
        u = nearest_graph_node(G, u_coord[0], u_coord[1])
        v = nearest_graph_node(G, v_coord[0], v_coord[1])
        if u != v:
            path = nx.shortest_path(G, u, v, weight='travel_time')
            edges.extend(zip(path[:-1], path[1:]))
    return edges

def route_length_km(G, edges):
    '''Total length of a route in km.'''
    return sum(G[u][v][0]['length'] for u, v in edges) / 1000.0

def route_cycle_time_min(G, edges, avg_speed_kmh=25):
    '''Estimated one-loop cycle time in minutes.'''
    length_km = route_length_km(G, edges)
    return (length_km / avg_speed_kmh) * 60
