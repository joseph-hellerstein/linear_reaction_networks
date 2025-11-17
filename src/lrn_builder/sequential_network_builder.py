'''Builds a sequential network with a degradation reaction at each stage.'''

def makeSequentialAntimony(num_stage: int) -> str:
    """
    Generates an Antimony file consisting of num_stage sequences of uni-uni reactions 
    and a degradation reaction. The kinetic constants have sequential values and are 
    labelled "k1", "k2", "k3", etc. with values 1, 2, 3, etc.
    
    Args:
        num_stage: Number of stages in the sequential network
        
    Returns:
        str: Antimony model string
        
    Example:
        For num_stage=3:
        - S0 -> S1; k1*S0   (k1 = 1)
        - S1 -> ; k2*S1     (k2 = 2)
        - S1 -> S2; k3*S1   (k3 = 3)
        - S2 -> ; k4*S2     (k4 = 4)
        - S2 -> S3; k5*S2   (k5 = 5)
        - S3 -> ; k6*S3     (k6 = 6)
    """
    if num_stage < 1:
        raise ValueError("num_stage must be at least 1")
    
    lines = []
    lines.append("model *sequential_network()")
    
    # Generate reactions and parameters for each stage
    k_counter = 1
    for i in range(num_stage):
        source_species = f"S{i}"
        target_species = f"S{i+1}"
        
        # Uni-uni reaction: Si -> Si+1
        forward_rate_name = f"k{k_counter}"
        lines.append(f"    {source_species} -> {target_species}; {forward_rate_name}*{source_species}")
        k_counter += 1
        
        # Degradation reaction: Si -> ;
        deg_rate_name = f"k{k_counter}"
        lines.append(f"    {source_species} -> ; {deg_rate_name}*{source_species}")
        k_counter += 1
    
    lines.append("")
    
    # Define kinetic constants
    for i in range(1, k_counter):
        lines.append(f"    k{i} = {i}")
    
    lines.append("")
    
    # Initialize species concentrations
    for i in range(num_stage + 1):
        lines.append(f"    S{i} = 0")
    
    lines.append("end")
    
    return "\n".join(lines)
