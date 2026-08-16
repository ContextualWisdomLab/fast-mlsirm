import re

with open('python/fast_mlsirm/estimators/marginal.py', 'r') as f:
    content = f.read()

# 1. Add MAX_QMC_XI_POINTS
content = content.replace('MAX_MARGINAL_WORKING_SET = 100_000_000\nMAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024',
                          'MAX_MARGINAL_WORKING_SET = 100_000_000\nMAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024\nMAX_QMC_XI_POINTS = 1_000_000')

# 2. Add validation in _preflight_xi_node_count
preflight_replacement = """        if type(xi_points) is not int or isinstance(xi_points, bool):
            raise ValueError("xi_points must be an exact built-in integer")
        if xi_points < 1:
            raise ValueError("xi_points must be a positive integer")
        if xi_points > MAX_QMC_XI_POINTS:
            raise ValueError(f"xi_points exceeds maximum limit of {MAX_QMC_XI_POINTS}")
        return xi_points"""

content = content.replace("""        if type(xi_points) is not int or isinstance(xi_points, bool):
            raise ValueError("xi_points must be an exact built-in integer")
        if xi_points < 1:
            raise ValueError("xi_points must be a positive integer")
        return xi_points""", preflight_replacement)

# 3. Add validation in _xi_nodes for QMC/Halton
qmc_replacement = """    if rule in {"qmc", "halton"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        if xi_points > MAX_QMC_XI_POINTS:
            raise ValueError(f"xi_points exceeds maximum limit of {MAX_QMC_XI_POINTS}")"""

content = content.replace("""    if rule in {"qmc", "halton"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")""", qmc_replacement)

# 4. Add validation in _xi_nodes for MC
mc_replacement = """    if rule in {"mc", "montecarlo", "monte-carlo"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")
        if xi_points > MAX_QMC_XI_POINTS:
            raise ValueError(f"xi_points exceeds maximum limit of {MAX_QMC_XI_POINTS}")"""

content = content.replace("""    if rule in {"mc", "montecarlo", "monte-carlo"}:
        if xi_points < 1:
            raise ValueError("xi_points must be >= 1 for the Halton/MonteCarlo rules")""", mc_replacement)

with open('python/fast_mlsirm/estimators/marginal.py', 'w') as f:
    f.write(content)
