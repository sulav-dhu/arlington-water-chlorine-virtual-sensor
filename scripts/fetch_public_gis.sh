#!/usr/bin/env bash
# Download the public City of Arlington GIS layers used to geocode sampling sites and derive
# location features. Source: https://gis2.arlingtontx.gov/agsext2/rest/services (public server).
#
# Each layer is paged (the server returns at most 2000 records per request) into
# <out>/pages/<layer>_<offset>.json; expected record counts go to <out>/counts.json.
# Uses curl because the server rejects Python's default HTTP client.
#
# Usage: scripts/fetch_public_gis.sh [out_dir]   (default: data/external/arlington_gis)
set -euo pipefail

OUT="${1:-data/external/arlington_gis}"
BASE="https://gis2.arlingtontx.gov/agsext2/rest/services"
mkdir -p "$OUT/pages"
rm -f "$OUT"/pages/*.json
echo "{" > "$OUT/counts.json"

fetch_layer() {
  local name="$1" url="$2" fields="$3" sep="$4"
  local count
  count=$(curl -sf --max-time 60 "$url/query?where=1%3D1&returnCountOnly=true&f=json" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["count"])')
  echo "$name: $count records"
  printf '  "%s": %s%s\n' "$name" "$count" "$sep" >> "$OUT/counts.json"
  for ((off = 0; off < count; off += 2000)); do
    curl -sf --max-time 120 -o "$OUT/pages/${name}_$(printf %06d "$off").json" \
      "$url/query?where=1%3D1&outFields=$fields&outSR=4326&orderByFields=OBJECTID&resultOffset=$off&resultRecordCount=2000&f=json"
    sleep 0.3
  done
}

fetch_layer address_points "$BASE/WaterUtilities/LCRRWaterServiceLineInventory/MapServer/1" \
  "OBJECTID,FullAddress" ","
fetch_layer hydrants "$BASE/Fire/Fire/MapServer/0" \
  "OBJECTID,PressurePlane,PressureSubnetworkName,AcceptanceDate,MfgDate,FlushProgram,OwnershipType,Easement" ","
fetch_layer road_centerlines "$BASE/OpenData/OD_Transportation/MapServer/1" \
  "OBJECTID,fullname,fromleft,toleft,fromright,toright,roadclass" ""

echo "}" >> "$OUT/counts.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/FETCHED_AT"
echo "Done -> $OUT"
