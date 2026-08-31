# SPDX-License-Identifier: Apache-2.0
"""Semantic cache identities for blockwise SFA transfers, including MTP.

Wire row numbers are transport-local: group ordering and draft cache names
can assign different rows on Prefill and Decode. Keep the existing address
arrays and describe which physical layer/component each slice belongs to.
"""

from collections.abc import Mapping

DsaCacheLayout = dict[str, tuple[int, int, int]]
DsaAddressArrays = tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]


def infer_dsa_block_group_ids(layer_names_by_group: list[list[str]]) -> dict[str, int]:
    groups: dict[str, set[int]] = {"main": set(), "indexer": set()}
    for group_id, names in enumerate(layer_names_by_group):
        for name in names:
            groups["indexer" if "indexer" in name.lower() else "main"].add(group_id)
    if len(groups["main"]) != 1 or len(groups["indexer"]) > 1:
        raise ValueError(
            "Blockwise DSA requires one manager group per cache component, including MTP; "
            "separate draft manager groups need per-layer block-ID routing"
        )
    main_group = next(iter(groups["main"]))
    return {"main": main_group, "indexer": next(iter(groups["indexer"]), main_group)}


def select_dsa_block_groups(
    remote_groups: tuple[tuple[int, ...], ...],
    group_ids: Mapping[str, int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return Main/Indexer block IDs without assuming manager group order."""
    if not remote_groups:
        raise ValueError("remote_block_ids must contain at least one group")
    if group_ids is None:
        # Preserve the old target-only peer convention. New Prefill publishes
        # the actual group IDs, even when diagnostic logging is disabled.
        return remote_groups[-1], remote_groups[0]
    if not isinstance(group_ids, Mapping) or set(group_ids) != {"main", "indexer"}:
        raise ValueError("DSA block group IDs must define main and indexer")
    for value in group_ids.values():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(remote_groups):
            raise ValueError("DSA block group ID is outside remote_block_ids")
    return remote_groups[group_ids["main"]], remote_groups[group_ids["indexer"]]


def dsa_cache_key(layer_name: str, num_target_layers: int) -> str:
    parts = layer_name.split(".")
    try:
        if "layers" in parts:
            layer = int(parts[parts.index("layers") + 1])
        elif "mtp" in parts:
            layer = num_target_layers + int(parts[parts.index("mtp") + 1])
        else:
            raise ValueError("missing physical layer index")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Blockwise DSA cannot identify physical layer in {layer_name!r}") from exc
    if layer < 0:
        raise ValueError("Blockwise DSA physical layer index must be nonnegative")
    component = "indexer" if "indexer" in layer_name.lower() else "main"
    return f"{component}:{layer}"


def add_dsa_cache_descriptor(
    layout: DsaCacheLayout,
    *,
    layer_name: str,
    num_target_layers: int,
    wire_layer: int,
    first_position: int,
    tensor_count: int,
) -> None:
    """Describe the exact appended tensor positions without reordering them."""
    key = dsa_cache_key(layer_name, num_target_layers)
    if key.startswith("indexer:"):
        if tensor_count not in (1, 2):
            raise ValueError("Blockwise DSA Indexer requires a key cache and optional scale cache")
        descriptors = {key: (wire_layer, first_position, tensor_count)}
    else:
        if tensor_count not in (2, 3, 4):
            raise ValueError("Blockwise DSA Main requires K/V and optional colocated Indexer caches")
        descriptors = {key: (wire_layer, first_position, 2)}
        if tensor_count > 2:
            indexer_key = key.replace("main:", "indexer:", 1)
            descriptors[indexer_key] = (wire_layer, first_position + 2, tensor_count - 2)
    duplicates = layout.keys() & descriptors.keys()
    if duplicates:
        raise ValueError(f"Duplicate blockwise DSA physical cache identities: {sorted(duplicates)}")
    layout.update(descriptors)


def project_dsa_remote_arrays(
    local_layout: list[list[tuple[int, int, int, int, int]]],
    local_cache_keys: Mapping[int, str],
    remote_cache_layout: Mapping[str, tuple[int, int, int]] | None,
    remote_arrays: DsaAddressArrays,
    *,
    num_target_layers: int,
) -> DsaAddressArrays:
    """Project remote cache slices onto this rank's local transport rows.

    Older peers remain usable for the existing target-only positional path.
    MTP needs explicit cache identities; a missing draft cache must never be
    mistaken for a shared target Indexer and silently skipped.
    """
    active_keys = {row: local_cache_keys[row] for row, entries in enumerate(local_layout) if entries}
    if remote_cache_layout is None:
        if any(int(key.split(":")[1]) >= num_target_layers for key in active_keys.values()):
            raise ValueError(
                "Blockwise DSA MTP requires cache descriptors from Prefill; update both P and D connectors"
            )
        return remote_arrays

    projected: DsaAddressArrays = tuple([[] for _ in local_layout] for _ in remote_arrays)
    for local_row, key in active_keys.items():
        descriptor = remote_cache_layout.get(key)
        if descriptor is None:
            component, layer_text = key.split(":")
            # GLM shared target Indexers can be absent on Prefill, but the
            # corresponding Main must exist. Draft Indexers are required.
            if component == "indexer" and int(layer_text) < num_target_layers:
                if f"main:{layer_text}" in remote_cache_layout:
                    continue
            raise ValueError(f"Blockwise DSA Prefill is missing required cache {key}")
        if len(descriptor) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in descriptor
        ):
            raise ValueError(f"Invalid blockwise DSA descriptor for {key}")
        remote_row, first, count = descriptor
        if count != len(local_layout[local_row]):
            raise ValueError(
                f"Blockwise DSA cache component count mismatch for {key}: "
                f"local={len(local_layout[local_row])} remote={count}"
            )
        for array, output in zip(remote_arrays, projected):
            if remote_row >= len(array) or first + count > len(array[remote_row]):
                raise ValueError(f"Incomplete blockwise DSA handshake arrays for {key}")
            output[local_row] = array[remote_row][first : first + count]
    return projected
