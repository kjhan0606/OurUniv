from hong2021_v65_structure_factorization_audit import (
    causal_flags,
    classify,
    donor_mapping,
    gradient_summary,
)


DOMAINS = ("TNG100", "SIMBA", "Swift")


def _minimal_v35() -> dict:
    return {
        "development_domains": {
            domain: {"train_objects": 20} for domain in DOMAINS
        }
    }


def test_v65_donor_mapping_is_deterministic_and_excludes_query() -> None:
    queries = {domain: [3, 7] for domain in DOMAINS}
    first = donor_mapping(_minimal_v35(), queries, 6, 123, same_domain=False)
    second = donor_mapping(_minimal_v35(), queries, 6, 123, same_domain=False)
    assert first == second
    for domain in DOMAINS:
        for query in first[domain]:
            for donor in query["members"]:
                if donor["donor_domain"] == domain:
                    assert donor["donor_index"] != query["query_object_index"]


def test_v65_classification_selects_only_frozen_causal_branches() -> None:
    direct = classify(True, False, True, True)
    joint = classify(True, True, True, True)
    copula = classify(True, True, False, False)
    none = classify(True, False, False, True)
    failed = classify(False, True, True, True)
    assert direct[2] is True and "direct_pair" in direct[1]
    assert joint[2] is True and "joint" in joint[1]
    assert copula[2] is True and "domain_conditioned_copula" in copula[1]
    assert none[2] is False
    assert failed[2] is False


def test_v65_gradient_summary_and_causal_rules_are_predeclared() -> None:
    rows = []
    for index, domain in enumerate(DOMAINS * 4):
        rows.append(
            {
                "domain": domain,
                "pair_gradient": [1.0, 0.1 + 0.001 * index],
                "bounded_NLL_gradient": [0.5, 0.05],
            }
        )
    gradients = gradient_summary(rows)
    summaries = {
        domain: {
            "source_balanced": {
                "median_mean_absolute_log_error": 1.0,
                "q90_mean_absolute_log_error": 1.2,
            },
            "same_domain": {
                "median_mean_absolute_log_error": 0.8,
                "q90_mean_absolute_log_error": 1.1,
            },
            "spatially_permuted_rank": {
                "median_mean_absolute_log_error": 1.1,
                "q90_mean_absolute_log_error": 1.3,
            },
            "rolled_parameter": {
                "median_mean_absolute_log_error": 1.2,
                "q90_mean_absolute_log_error": 1.4,
            },
        }
        for domain in DOMAINS
    }
    assert gradients["global_median_leave_one_out_cosine"] > 0.99
    assert causal_flags(summaries, gradients) == (True, True, True)
