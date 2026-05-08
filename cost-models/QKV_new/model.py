def cost(path: str):
    # Define constant costs for each instruction
    instruction_costs = {
        'load_01': 20,
        'load_03': 40,
        'store_10': 20,
        'store_30': 40,
        'transpose_13': 10,
        'mov_21': 5,
        'mov_23': 5,
        'gemm_33': 100,
        'gemm_13': 80,
        'softmax': 50
    }
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    total_cost = 0
    in_qkv_function = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def qkv_():'):
            in_qkv_function = True
            continue
        elif in_qkv_function and stripped.startswith('api.'):
            # Extract instruction name
            instr = stripped.split('.')[1].split('(')[0]
            if instr in instruction_costs:
                total_cost += instruction_costs[instr]
        elif in_qkv_function and stripped == 'return qkv_':
            break
    
    return total_cost
