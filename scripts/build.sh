#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
ns3_dir=${1:?Usage: bash scripts/build.sh /absolute/path/to/ns-3.32}
profile=${NS3_BUILD_PROFILE:-debug}

test -f "$ns3_dir/build/ns3/core-module.h" || { echo "ns-3.32 build headers not found" >&2; exit 1; }
test -d "$ns3_dir/build/lib" || { echo "ns-3.32 build libraries not found" >&2; exit 1; }
mkdir -p "$repo_dir/build"
modules=(wifi internet mobility propagation spectrum antenna energy network core stats traffic-control bridge)
libs=()
for module in "${modules[@]}"; do
  libs+=("-lns3.32-$module-$profile")
done
for source in p2_native_density p2_native_kf15 p2_native_absolute_speed; do
  g++ -std=c++17 -O2 -I"$ns3_dir/build" "$repo_dir/src/${source}.cc" \
    -L"$ns3_dir/build/lib" -Wl,--no-as-needed "${libs[@]}" \
    -Wl,-rpath,"$ns3_dir/build/lib" -o "$repo_dir/build/${source}"
  echo "Built $repo_dir/build/${source}"
done
