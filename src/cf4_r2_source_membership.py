"""Source membership states, not physical isolation or a membership prior."""


def membership_state(source_group, source_richness, matched_group=None,
                     matched_richness=None):
    """Keep absent-catalogue objects distinct from catalogue singletons.

    Howlett+2022 sec.2.2 assigns no group to missing Tempel objects. This
    is not evidence that an object has no physical companions. Never
    attach a missing grouped object to a group by proximity as a fallback.
    """
    if source_group < 0 or source_richness < 1:
        raise ValueError('invalid published membership')
    if matched_group is None:
        return ('source_ungrouped_catalogue_absent' if source_group == 0
                and source_richness == 1 else 'unresolved_missing_group_member')
    if matched_group != source_group or matched_richness != source_richness:
        return 'unresolved_membership_conflict'
    if source_group == 0:
        if source_richness != 1:
            return 'unresolved_membership_conflict'
        return 'source_ungrouped_catalogue_present'
    if source_richness < 2:
        return 'unresolved_membership_conflict'
    return 'source_grouped_catalogue_present'
