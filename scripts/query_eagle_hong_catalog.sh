#!/usr/bin/env bash
# Query the EAGLE galaxy sample used by the Hong et al. independent test.

set -euo pipefail

root=${EAGLE_ROOT:-/gpfs/kjhan/EAGLE/RefL0100N1504}
netrc=${VIRGO_NETRC:-$HOME/.config/virgodb/netrc}
endpoint=http://virgodb.dur.ac.uk:8080/Eagle/MyDB
catalog_dir=$root/catalogs
catalog=$catalog_dir/RefL0100N1504_Hong_targets_snap28.csv
cookie=$catalog_dir/.virgodb_hong.cookies

# TNG100 has 48,296 M_B<-15 targets in (75 Mpc/h)^3.  Matching that
# number density in the (67.77 Mpc/h)^3 EAGLE box gives 35,631.97, rounded
# to the nearest integer as frozen here.
expected_rows=35632

if [[ ! -r "$netrc" ]]; then
    printf 'Missing VirgoDB credential file: %s\n' "$netrc" >&2
    exit 2
fi

mkdir -p "$catalog_dir"
chmod 700 "$catalog_dir"

# Hong et al. explicitly use a stellar-mass rank cut for EAGLE rather than
# EAGLE photometry.  The 30-pkpc aperture mass is the EAGLE-recommended
# stellar-mass definition.  The secondary GalaxyID ordering freezes ties.
sql='select top 35632 s.GalaxyID,s.GroupNumber,s.SubGroupNumber,s.CentreOfPotential_x,s.CentreOfPotential_y,s.CentreOfPotential_z,s.Velocity_x,s.Velocity_y,s.Velocity_z,a.Mass_Star as Mstar_30pkpc from Eagle..RefL0100N1504_SubHalo as s join Eagle..RefL0100N1504_Aperture as a on s.GalaxyID=a.GalaxyID where s.SnapNum=28 and s.Spurious=0 and a.ApertureSize=30 and a.Mass_Star>0 order by a.Mass_Star desc,s.GalaxyID asc'

if [[ -e "$catalog" ]]; then
    printf 'Refusing to overwrite existing catalogue: %s\n' "$catalog" >&2
    exit 3
fi

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
    exit 4
fi

header='GalaxyID,GroupNumber,SubGroupNumber,CentreOfPotential_x,CentreOfPotential_y,CentreOfPotential_z,Velocity_x,Velocity_y,Velocity_z,Mstar_30pkpc'
if ! grep -Fxq "$header" "$catalog.partial"; then
    printf 'VirgoDB catalogue header does not match the frozen schema.\n' >&2
    exit 5
fi

data_rows=$(grep -vc '^#' "$catalog.partial")
data_rows=$((data_rows - 1))
if [[ "$data_rows" -ne "$expected_rows" ]]; then
    printf 'Catalogue row mismatch: %s != %s\n' \
        "$data_rows" "$expected_rows" >&2
    exit 6
fi

mv "$catalog.partial" "$catalog"
chmod 600 "$catalog"
rm -f "$cookie"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$data_rows" "$catalog" \
    >"$root/query_ref100_hong.complete"
printf '[complete] rows=%s catalog=%s\n' "$data_rows" "$catalog"
