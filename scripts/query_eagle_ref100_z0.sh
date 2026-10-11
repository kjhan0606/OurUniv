#!/usr/bin/env bash
# Download the z=0 RefL0100N1504 galaxy catalogue needed for Hong-style cubes.

set -euo pipefail

root=${EAGLE_ROOT:-/gpfs/kjhan/EAGLE/RefL0100N1504}
netrc=${VIRGO_NETRC:-$HOME/.config/virgodb/netrc}
endpoint=http://virgodb.dur.ac.uk:8080/Eagle/MyDB
catalog_dir=$root/catalogs
catalog=$catalog_dir/RefL0100N1504_SubHalo_Magnitudes_snap28.csv
expected_rows=29737
cookie=$catalog_dir/.virgodb.cookies

if [[ ! -r "$netrc" ]]; then
    printf 'Missing VirgoDB credential file: %s\n' "$netrc" >&2
    exit 2
fi

mkdir -p "$catalog_dir"
chmod 700 "$catalog_dir"

sql='select s.GalaxyID,s.GroupNumber,s.SubGroupNumber,s.CentreOfPotential_x,s.CentreOfPotential_y,s.CentreOfPotential_z,s.Velocity_x,s.Velocity_y,s.Velocity_z,s.MassType_Star,s.MassType_DM,m.u_nodust,m.g_nodust,m.r_nodust,m.i_nodust,m.z_nodust from Eagle..RefL0100N1504_SubHalo as s join Eagle..RefL0100N1504_Magnitudes as m on s.GalaxyID=m.GalaxyID where s.SnapNum=28 and m.SnapNum=28'

curl --fail --show-error --silent \
    --netrc-file "$netrc" \
    --cookie "$cookie" --cookie-jar "$cookie" \
    --data-urlencode 'action=doQuery' \
    --data-urlencode 'queryMode=stream' \
    --data-urlencode 'MAXROWS=-1' \
    --data-urlencode "SQL=$sql" \
    "$endpoint" >"$catalog.partial"

if ! head -n 1 "$catalog.partial" | grep -Fxq '#OK'; then
    printf 'VirgoDB query did not return #OK:\n' >&2
    head -n 20 "$catalog.partial" >&2
    exit 3
fi

header='GalaxyID,GroupNumber,SubGroupNumber,CentreOfPotential_x,CentreOfPotential_y,CentreOfPotential_z,Velocity_x,Velocity_y,Velocity_z,MassType_Star,MassType_DM,u_nodust,g_nodust,r_nodust,i_nodust,z_nodust'
if ! grep -Fxq "$header" "$catalog.partial"; then
    printf 'VirgoDB catalogue header does not match the frozen schema.\n' >&2
    exit 4
fi

data_rows=$(grep -vc '^#' "$catalog.partial")
data_rows=$((data_rows - 1))
if [[ "$data_rows" -ne "$expected_rows" ]]; then
    printf 'Catalogue row mismatch: %s != %s\n' "$data_rows" "$expected_rows" >&2
    exit 5
fi

mv -f "$catalog.partial" "$catalog"
chmod 600 "$catalog"
rm -f "$cookie"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$data_rows" "$catalog" \
    >"$root/query_ref100_z0.complete"
printf '[complete] rows=%s catalog=%s\n' "$data_rows" "$catalog"
