"""
data_parser.py
==============
Parses the raw digitized field-survey CSV files (DATA-*.csv) and writes a clean, normalized
boarding_counts.csv that the demand module can consume.

Field-survey CSV schema (per file):
  Each file contains MULTIPLE time-window sections separated by blank rows.
  Each section has:
    Row 0: Node name  (e.g. "ROTUNDA", "SM")
    Row 1: Date       (e.g. "August 14, 2026")
    Row 2: Time window (e.g. "6:00 AM - 8:00 AM" / "11:00 AM - 1:00 PM" / "4:00 PM - 6:00 PM")
    Row 3: Header     ("Time", "Jeepney Route", "Stopped", "Passengers", "Fullness", "Waiting")
    Row 4+: Observations (until the next blank row or end of file)

Output CSV columns:
  date, node, window, route, passengers_boarding, obs_duration_min
"""

import csv
import glob
import os
import re

# ---------------------------------------------------------------------------
# Canonical identifiers used by the model
# ---------------------------------------------------------------------------
NODES = [
    'ROTUNDA',          # index 0 - Ayala / Rotunda
    'PACIFIC MALL',     # index 1
    'DARAGA MARKET',    # index 2
    'EMBARCADERO',      # index 3
    'SM',               # index 4
]

ROUTES = [
    'DIRETSO A',        # index 0  (LPTRP route 11, via Lapu-Lapu St)
    'DIRETSO B',        # index 1  (LPTRP route 12, via Capitol)
    'RAWIS A',          # index 2  (LPTRP route 13)
    'RAWIS B',          # index 3  (LPTRP route 14)
    'ARIMBAY',          # index 4  (LPTRP route 15, via Tahao Rd)
    'LOOP 1',           # index 5  (LPTRP route 16-1)
    'LOOP 2',           # index 6  (LPTRP route 16-2)
    'EXTERNAL',         # index 7  (Camalig, Guinobatan, Ligao, etc.)
]

WINDOWS = ['AM', 'MID', 'PM']

# ---------------------------------------------------------------------------
# Normalization tables
# ---------------------------------------------------------------------------
_NODE_MAP = {
    'ROTUNDA': 'ROTUNDA',
    'AYALA': 'ROTUNDA',
    'RIZAL': 'ROTUNDA',
    'PACIFIC': 'PACIFIC MALL',
    'PACIFIC MALL': 'PACIFIC MALL',
    'DARAGA': 'DARAGA MARKET',
    'DARAGA PUBLIC MARKET': 'DARAGA MARKET',
    'DARAGA MARKET': 'DARAGA MARKET',
    'EMBARCADERO': 'EMBARCADERO',
    'EMBARCDERO': 'EMBARCADERO',   # typo present in raw data
    'EMBARCADRO': 'EMBARCADERO',   # alternate typo
    'SM': 'SM',
}

_ROUTE_MAP = {
    'DIRETSO': 'DIRETSO A',
    'DITETSO': 'DIRETSO A',       # known typo in raw data
    'DIRETSO A': 'DIRETSO A',
    'ALTERNATE': 'DIRETSO B',
    'DIRETSO B': 'DIRETSO B',
    'RAWIS A': 'RAWIS A',
    'RAWIS B': 'RAWIS B',
    'RAWIS': 'RAWIS A',
    'ARIMBAY': 'ARIMBAY',
    'TAHAO': 'ARIMBAY',
    'TAHAO ROAD': 'ARIMBAY',
    'L1': 'LOOP 1',
    'LOOP 1': 'LOOP 1',
    'L2': 'LOOP 2',
    'LOOP 2': 'LOOP 2',
    'CAMALIG': 'EXTERNAL',
    'GUINOBATAN': 'EXTERNAL',
    'LIGAO': 'EXTERNAL',
    'OAS': 'EXTERNAL',
    'POLANGUI': 'EXTERNAL',
    'TABACO': 'EXTERNAL',
    'STO DOMINGO': 'EXTERNAL',
    'MALABOG': 'EXTERNAL',
}

# Time-window patterns: maps header strings → canonical window label.
# The raw data uses several formats:
#   AM:  "6:00 AM - 8:00 AM", "6:00 AM – 8:00 AM"
#   MID: "11:00 AM - 1:00 PM", "11:00 AM – 1:00 PM"
#   PM:  "4:00 PM - 6:00 PM",  "4:00 PM – 6:00 PM", "4:00 – 4:15 PM" (sub-slot, same window)
_WINDOW_PATTERNS = [
    # AM: any window starting around 5-8 AM
    (re.compile(r'\b(5|6|7)\b.*?am', re.I), 'AM'),
    # MID: any window including 11 AM or noon
    (re.compile(r'\b11\b', re.I), 'MID'),
    (re.compile(r'11\s*:\s*00', re.I), 'MID'),
    # PM: any window starting around 3-5 PM
    (re.compile(r'\b(3|4|5)\b.*?pm', re.I), 'PM'),
    (re.compile(r'(4|5)\s*:\s*00.*pm', re.I), 'PM'),
    # Fallback hour-range patterns
    (re.compile(r'6.*?[–\-].*?8', re.I), 'AM'),
    (re.compile(r'11.*?[–\-].*?(1|13)', re.I), 'MID'),
    (re.compile(r'4.*?[–\-].*?6', re.I), 'PM'),
]

_MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


def _normalize_node(raw):
    key = re.sub(r'\s+', ' ', raw.strip()).upper()
    return _NODE_MAP.get(key)


def _normalize_route(raw):
    key = re.sub(r'\s+', ' ', raw.strip()).upper()
    if key in _ROUTE_MAP:
        return _ROUTE_MAP[key]
    # Partial prefix match for long LPTRP-style names in data
    for pattern, canonical in _ROUTE_MAP.items():
        if key.startswith(pattern.upper()):
            return canonical
    return None


def _normalize_window(raw):
    raw_clean = raw.strip()
    for pat, label in _WINDOW_PATTERNS:
        if pat.search(raw_clean):
            return label
    upper = raw_clean.upper()
    if upper in WINDOWS:
        return upper
    return None


def _parse_date(raw):
    """Return ISO date string (YYYY-MM-DD) or None."""
    raw = raw.strip()
    for month_name, month_num in _MONTH_MAP.items():
        if month_name in raw.lower():
            nums = re.findall(r'\d+', raw)
            if len(nums) >= 2:
                day = int(nums[0])
                year = int(nums[-1])
                return '{:04d}-{:02d}-{:02d}'.format(year, month_num, day)
    return None


def _is_section_header(line):
    """
    Detect the start of a new time-window section.
    A section starts when the first cell looks like a node name or a date.
    Returns True if this line looks like a node/section-header row.
    """
    if not line or not line[0].strip():
        return False
    first = line[0].strip().upper()
    # Node names
    if first in _NODE_MAP:
        return True
    # Date lines start with a month name
    for month in _MONTH_MAP:
        if month in first.lower():
            return True
    return False


def _looks_like_time_window(text):
    """Return True if text looks like a time-window declaration."""
    text = text.strip()
    return bool(re.search(r'\d+\s*:\s*\d+', text)) and bool(
        re.search(r'(am|pm|\d+\s*:\s*00)', text, re.I)
    )


def _parse_file(filepath):
    """
    Parse a single field-survey CSV.
    Each file may contain multiple sections (AM / MID / PM) separated by
    blank rows. Each section repeats the 4-row header block.
    Returns a list of observation dicts.
    """
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        reader = csv.reader(fh)
        lines = list(reader)

    # ── Split into sections ────────────────────────────────────────────────
    # A new section starts whenever we see a non-empty node name row,
    # which repeats the [node, date, window, header] 4-row prefix.
    sections = []
    current = []
    for line in lines:
        first_cell = line[0].strip() if line else ''
        # blank row between sections
        if not any(c.strip() for c in line):
            if current:
                sections.append(current)
                current = []
            continue
        current.append(line)
    if current:
        sections.append(current)

    # ── Parse each section independently ──────────────────────────────────
    for section in sections:
        if len(section) < 5:
            continue

        # The section header is always: [node, date, window, col-header, data...]
        node_raw = section[0][0] if section[0] else ''
        date_raw = section[1][0] if section[1] else ''
        win_raw  = section[2][0] if section[2] else ''

        node   = _normalize_node(node_raw)
        date   = _parse_date(date_raw)
        window = _normalize_window(win_raw)

        if not node or not date or not window:
            continue

        obs_duration_min = 15  # each time slot = 15-minute block

        # Data rows start after the column-header row (index 3)
        for line in section[4:]:
            if len(line) < 2:
                continue

            route_raw = line[1].strip() if len(line) > 1 else ''
            stopped   = line[2].strip().upper() if len(line) > 2 else ''
            pax_raw   = line[3].strip() if len(line) > 3 else ''

            # Normalize em-dash placeholders used in some sheets
            pax_raw = pax_raw.replace('—', '').replace('–', '').strip()

            # Only count jeepneys that actually stopped to board
            if stopped != 'Y':
                continue

            route = _normalize_route(route_raw)
            if route is None:
                continue

            try:
                passengers = int(pax_raw) if pax_raw else 0
            except ValueError:
                passengers = 0

            rows.append({
                'date': date,
                'node': node,
                'window': window,
                'route': route,
                'passengers_boarding': passengers,
                'obs_duration_min': obs_duration_min,
            })

    return rows


def build_boarding_counts(
    raw_dir=None,
    out_path='data/processed/boarding_counts.csv',
):
    """
    Read all DATA-*.csv survey files from raw_dir and write a consolidated
    boarding_counts.csv to out_path (compatible with src/demand.py).
    """
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    pattern = os.path.join(str(raw_dir), 'DATA-*.csv')
    files = sorted(glob.glob(pattern))

    # Skip non-survey files
    skip_keywords = ['LPTRP', 'REGISTERED']
    files = [
        f for f in files
        if not any(kw in os.path.basename(f).upper() for kw in skip_keywords)
    ]

    all_rows = []
    for f in files:
        parsed = _parse_file(f)
        print('  {:40s} -> {} observations'.format(os.path.basename(f), len(parsed)))
        all_rows.extend(parsed)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = ['date', 'node', 'window', 'route', 'passengers_boarding', 'obs_duration_min']
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print('\nWrote {} rows -> {}'.format(len(all_rows), out_path))

    # Print window coverage summary
    window_totals = {'AM': 0, 'MID': 0, 'PM': 0}
    for r in all_rows:
        window_totals[r['window']] = window_totals.get(r['window'], 0) + r['passengers_boarding']
    print('\nPassenger totals by time window (from parsed data):')
    for w, total in window_totals.items():
        print('  {:5s}: {:,} pax'.format(w, total))

    return all_rows


if __name__ == '__main__':
    build_boarding_counts()
