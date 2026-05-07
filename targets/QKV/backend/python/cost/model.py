def cost(path: str):
    # Define constant costs for each instruction
    instruction_costs = {
        'load_rm': 10,
        'load_cm': 10,
        'store_rm': 10,
        'store_cm': 10,
        'mov': 5,
        'gemm': 100,
        'softmax': 50
    }
    
    total_cost = 0
    with open(path, 'r') as f:
        lines = f.readlines()
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
        
