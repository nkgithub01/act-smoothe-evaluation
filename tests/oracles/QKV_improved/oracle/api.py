import os, sys
import jax.numpy as jnp
import ast
from .utils import *
from collections import defaultdict

TARGET_NAME = "QKV_improved"

def semantic_init(hbm_size):
	global_counters = {
		'd0': {
			'type': f'u8[{hbm_size}]',
			'dim': '{0}',
			'counter': 0
		},
		'd1': {
			'type': 'bf16[128,64]',
			'dim': '{1,0}',
			'counter': 0
		},
		'd2': {
			'type': 'bf16[64,64]',
			'dim': '{1,0}',
			'counter': 0
		},

		'instruction_counter': 0,
		'start_counter': 0,
		'total_loop_counter': 0,
		'loop_counter': 0,
		'if_counter': 0,
		'loop_var': 0,
	}

	prologue = []
	prologue.append(f'%d0.0 = u8[{hbm_size}]{{0}} parameter(0)')

	prologue.append(f'%d1.0 = bf16[128,64]{{1,0}} constant(0)')
	prologue.append(f'%d2.0 = bf16[64,64]{{1,0}} constant(0)')

	return global_counters, prologue



default_state = {

}

state = {

}

custom_functions = """
add_bf16{
	%a = bf16[] parameter(0)
	%b = bf16[] parameter(1)
	ROOT %sum = bf16[] add(%a, %b)
}
add_s8{
	%a = s8[] parameter(0)
	%b = s8[] parameter(1)
	ROOT %sum = s8[] add(%a, %b)
}
add_s32{
	%a = s32[] parameter(0)
	%b = s32[] parameter(1)
	ROOT %sum = s32[] add(%a, %b)
}
"""

main_function_call = """
HloModule compiled\n""" + custom_functions + "\n"

global_counters = None
executable = None

mode = None
total_cost = 0
instruction_scopes = defaultdict(list)
instructions = []

loop_variables = ["main"]
conditional_clauses = []
cur_scope = "main"
scope_tuple_count = defaultdict(int)

instruction_store = []
compare_function = ""
call_statement = ""

def generate_full_hlotext(hbm_size: int):
	global instructions, main_function_call, compare_function, call_statement, global_counters, loop_variables, cur_scope, instruction_scopes
	counters, prologue = semantic_init(hbm_size)
	hlo_text = main_function_call + "\n"

	hlo_text += compare_function

	for scope, lines in reversed(list(instruction_scopes.items())):
		if(scope != "main"):
			for line in lines:
				hlo_text += "\t" + line + "\n"

	# Main call
	hlo_text += """\nENTRY %main """
	hlo_text += "{\n"
	for line in prologue:
		hlo_text += "\t" + line + "\n"
	for line in instruction_scopes["main"]:
		hlo_text += "\t" + line + "\n"
	hlo_text += f"\tROOT %d0.res = u8[{hbm_size}] copy(%d0.{global_counters['d0']['counter']})\n"
	hlo_text+= "}\n"

	return hlo_text

def start_loop(var_name, start, end):
	global instructions, instruction_store, compare_function, call_statement, global_counters, loop_variables, cur_scope, instruction_scopes
	instruction_store = instructions
	var_name = "%" + var_name
	instructions = []
	counter_types = "s32[]"
	num_loop_counter = global_counters['total_loop_counter']
	scope_tuple_count[cur_scope] += 1

	for name, counter in global_counters.items():
		if isinstance(counter, dict):
			counter_types += ', ' + counter['type']
	for var in loop_variables[1:]:
		counter_types += ', s32[]'
	compare_function += f"""
%compare_function{global_counters['total_loop_counter']}(current_vals: ({counter_types})) -> pred[]{{
	%comp_tuple = ({counter_types}) parameter(0)
	%current_index = s32[] get-tuple-element(%comp_tuple), index=0
	%target = s32[] constant({end})
	ROOT %continue = pred[] compare(%current_index, %target), direction=LT
}}
 """
	tuple_num = scope_tuple_count[cur_scope]
	initial_tuple = f"%initial_index.{tuple_num}"
	num_counters = 0
	for name, counter in global_counters.items():
		if isinstance(counter, dict):
			initial_tuple += f", %{name}.{counter['counter']}"
			num_counters += 1
	for var in loop_variables[1:]:
		initial_tuple += f", {var}"
	
	call_statement = f"""
	%initial_index.{tuple_num} = s32[] constant({start})
	  %init_tuple.{tuple_num} = tuple({initial_tuple})
	%current_tuple.{tuple_num} = ({counter_types}) while(%init_tuple.{tuple_num}), body=%while_body{global_counters['total_loop_counter']}, condition=%compare_function{global_counters['total_loop_counter']}
 """
	instruction_scopes[cur_scope].append(call_statement)
	
	while_prologue = f"""
%while_body{global_counters['total_loop_counter']}(current_vals: ({counter_types})) -> ({counter_types}){'{'}
	%start_tuple = ({counter_types}) parameter(0)
	{var_name} = s32[] get-tuple-element(%start_tuple), index=0\n
 """
	cur_count = 1
	for var in loop_variables[1:]:
		while_prologue += f"\t{var} = s32[] get-tuple-element(%start_tuple), index={cur_count + num_counters}\n"
		cur_count += 1
	
	cur_ind = 1
	for name, counter in global_counters.items():
		if isinstance(counter, dict):
			while_prologue += f"\t%{name}.{counter['counter']} = {counter['type']} get-tuple-element(%start_tuple), index={cur_ind}\n"
			cur_ind += 1
	cur_scope = var_name
	instruction_scopes[cur_scope].append(while_prologue)
	loop_variables.append(var_name)
	global_counters['total_loop_counter'] += 1

def end_loop():
	global loop_variables, global_counters, instruction_scopes, cur_scope
	var_name = loop_variables[-1]
	instruction_scopes[cur_scope].append(f"%new_loop_index = s32[] add({var_name}, s32[] constant(1))")
	ret_val = f"ROOT %result = tuple(%new_loop_index"
	for name, counter in global_counters.items():
		if isinstance(counter, dict):
			ret_val += f", %{name}.{counter['counter']}"
	for var in loop_variables[1:-1]:
		ret_val += f", {var}"
	ret_val += ')\n}'
	instruction_scopes[cur_scope].append(ret_val)
	loop_variables.pop()
	cur_scope = loop_variables[-1]
	cur_ind = 1
	for name, counter in global_counters.items():
		if isinstance(counter, dict):
			counter['counter'] += 1
			instruction_scopes[cur_scope].append(f"%{name}.{counter['counter']} = {counter['type']} get-tuple-element(%current_tuple.{scope_tuple_count[cur_scope]}), index={cur_ind}")
			cur_ind += 1

def debug(prefix: str, data: str) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	attrs = {}
	
	if mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
		

		prefix_bytes = jnp.frombuffer(prefix.encode('utf-8'), dtype=jnp.uint8)
		prefix_len = len(prefix_bytes)
		prefix_string = "{"
		for i in range(len(prefix_bytes)):
			if i != 0:
				prefix_string += ", "
			prefix_string += f"{prefix_bytes[i]}"
		prefix_string += "}"

		data_name = data.split("[")[0]

		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, data_name)
		data_dims = ast.literal_eval('[' + global_counters[data_name]["type"].split("[")[1].split("]")[0] + ']')
		debug_dims = ast.literal_eval('[' + data.split("[")[1].split("]")[0] + ']')
		slice_configs = []
		for i in range(len(debug_dims)):
			data_dims[i] = 1
		for i in range(len(data_dims)):
			data_dims[i] = str(data_dims[i])
			if i < len(debug_dims):
				slice_configs.append(str(debug_dims[i]))
			else:
				slice_configs.append('0')
		
		data_type = global_counters[data_name]["type"].split("[")[0]
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'debug_slice', data_type, data_dims)
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)

		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'debug_slice')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'debug_slice_f64', 'f64', data_dims)
		line = f'{lhs_loc} = {lhs_type} convert({rhs_loc})'
		output.append(line)

		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'debug_slice_f64')
		debug_prefix_loc = f'debug_prefix.{global_counters["instruction_counter"]}'
		output.append(f'{debug_prefix_loc} = u8[{prefix_len}] constant({prefix_string})')
		output.append(f'call.{global_counters["instruction_counter"]} = s32[] custom-call({debug_prefix_loc}, {rhs_loc}), custom_call_target="print_handler", custom_call_has_side_effect=true, api_version=API_VERSION_TYPED_FFI')
		instruction_scopes[cur_scope] += output
		global_counters['instruction_counter'] += 1

def load_rm(n,addr_in,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr_in": addr_in,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd0')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'u8', ['@c.n * 128'])
		slice_configs = ['@a.addr_in:@a.addr_in+@c.n * 128']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "a", "u8", ['@c.n',64,2])
		line = reshape_helper(lhs_loc, lhs_type, rhs_loc)
		output.append(line)
		lvars['a'] = lhs_loc
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "bf16", ['@c.n',64])
		line = f'{lhs_loc} = {lhs_type} bitcast-convert({rhs_loc})'
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd1')
		start_indices = ['@a.addr_out','0']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'load_rm(n,addr_in,addr_out)')
		print([n,addr_in,addr_out])

def load_cm(n,addr_in,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr_in": addr_in,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd0')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'u8', ['@c.n * 128'])
		slice_configs = ['@a.addr_in:@a.addr_in+@c.n * 128']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "a", "u8", ['@c.n',64,2])
		line = reshape_helper(lhs_loc, lhs_type, rhs_loc)
		output.append(line)
		lvars['a'] = lhs_loc
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "b", "bf16", ['@c.n',64])
		line = f'{lhs_loc} = {lhs_type} bitcast-convert({rhs_loc})'
		output.append(line)
		lvars['b'] = lhs_loc
		in_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "b")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "bf16", [64,'@c.n'])
		dims = "{1, 0}"
		dims = dims.replace("'","")
		line = f'{lhs_loc} = {lhs_type} transpose({in_loc}), dimensions={dims}'
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd1')
		start_indices = ['@a.addr_out','0']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'load_cm(n,addr_in,addr_out)')
		print([n,addr_in,addr_out])

def store_rm(n,addr_in,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr_in": addr_in,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'bf16', ['@c.n','64'])
		slice_configs = ['@a.addr_in:@a.addr_in+@c.n','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "a", "u8", ['@c.n',64,2])
		line = f'{lhs_loc} = {lhs_type} bitcast-convert({rhs_loc})'
		output.append(line)
		lvars['a'] = lhs_loc
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "u8", ['@c.n*128'])
		line = reshape_helper(lhs_loc, lhs_type, rhs_loc)
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd0')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd0')
		start_indices = ['@a.addr_out']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'store_rm(n,addr_in,addr_out)')
		print([n,addr_in,addr_out])

def store_cm(n,addr_in,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr_in": addr_in,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'bf16', ['@c.n','64'])
		slice_configs = ['@a.addr_in:@a.addr_in+@c.n','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		in_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "a", "bf16", [64,'@c.n'])
		dims = "{1, 0}"
		dims = dims.replace("'","")
		line = f'{lhs_loc} = {lhs_type} transpose({in_loc}), dimensions={dims}'
		output.append(line)
		lvars['a'] = lhs_loc
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "b", "u8", [64,'@c.n',2])
		line = f'{lhs_loc} = {lhs_type} bitcast-convert({rhs_loc})'
		output.append(line)
		lvars['b'] = lhs_loc
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "b")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "u8", ['@c.n*128'])
		line = reshape_helper(lhs_loc, lhs_type, rhs_loc)
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd0')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd0')
		start_indices = ['@a.addr_out']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'store_cm(n,addr_in,addr_out)')
		print([n,addr_in,addr_out])

def mov(n,addr_in,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr_in": addr_in,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd2')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'bf16', ['@c.n','64'])
		slice_configs = ['@a.addr_in:@a.addr_in+@c.n','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "bf16", ['@c.n',64])
		line = f'{lhs_loc} = {lhs_type} copy({rhs_loc})'
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd1')
		start_indices = ['@a.addr_out','0']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'mov(n,addr_in,addr_out)')
		print([n,addr_in,addr_out])

def gemm(addr_1,addr_2,addr_out) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {

	}
	attrs = {
			"addr_1": addr_1,
			"addr_2": addr_2,
			"addr_out": addr_out,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'bf16', ['64','64'])
		slice_configs = ['@a.addr_1:@a.addr_1+64','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)

		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd1')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In2', 'bf16', ['64','64'])
		slice_configs = ['@a.addr_2:@a.addr_2+64','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		a_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		b_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In2")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "bf16", [64,64])
		line = dot_helper(lhs_loc, lhs_type, a_loc, b_loc,
						  {}, {1}, {}, {0})
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd2')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd2')
		start_indices = ['@a.addr_out','0']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'gemm(addr_1,addr_2,addr_out)')
		print([addr_1,addr_2,addr_out])

def softmax(n,addr) -> None:
	global total_cost, instruction_scopes, cur_scope, comp_attrs, state, global_counters
	global_counters['loop_counter'] = 0
	comp_attrs = {
			"n": n,

	}
	attrs = {
			"addr": addr,

	}
	flag = 1

	assert(flag) 

	if mode == 'fsim':
		pass
	elif mode == 'fsim-compile':
		lvars = {}
		output = []
		for name, arg in attrs.items():
			if(isinstance(arg, str)):
				hlo_lines, new_name = generate_loop_eq_vars(arg, global_counters)
				output.append(hlo_lines)
				attrs[name] = new_name
				global_counters['loop_var'] += 1
	
		for name, arg in comp_attrs.items():
			if(isinstance(arg, str)):
				raise ValueError("Loop expressions are not supported for computation attributes")
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd2')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'In1', 'bf16', ['@c.n','64'])
		slice_configs = ['@a.addr:@a.addr+@c.n','0:64']
		line = slice_load(lhs_loc, lhs_type, rhs_loc, slice_configs, attrs, state, lvars, comp_attrs)
		output.append(line)


		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "In1")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "a", "bf16", ['@c.n',64])
		line = f'{lhs_loc} = {lhs_type} exponential({rhs_loc})'
		output.append(line)
		lvars['a'] = lhs_loc
		a_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		#b_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, {{B}})
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "reduced", "bf16", ['@c.n'])
		dim = {1}
		apply = "%add_bf16"
		dtype = lhs_type.split('[')[0]
		line = f'{lhs_loc} = {lhs_type} reduce({a_loc}, {dtype}[] constant(0)), dimensions={dim}, to_apply={apply}'
		output.append(line)
		lvars['reduced'] = lhs_loc
		b_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "reduced")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "b", "bf16", ['@c.n',64])
		dims = create_num_str(['@c.n',64])
		line = f"{lhs_loc} = {lhs_type} broadcast({b_loc}), dimensions={dim}"
		output.append(line)
		lvars['b'] = lhs_loc
		a_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "a")
		b_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, "b")
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, "Out0", "bf16", ['@c.n',64])
		line = f'{lhs_loc} = {lhs_type} divide({a_loc}, {b_loc})'
		output.append(line)
		lvars['Out0'] = lhs_loc
		rhs_update = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'Out0')
		rhs_loc = parameter_util(global_counters,  attrs, state, lvars, comp_attrs, 'd2')
		lhs_loc, lhs_type = lhs_util(attrs, state, global_counters, lvars, comp_attrs, 'd2')
		start_indices = ['@a.addr','0']
		lines = slice_store(lhs_loc, lhs_type, rhs_loc, rhs_update, start_indices,
							attrs, state, lvars, comp_attrs, global_counters['start_counter'])
		global_counters["start_counter"] += 1
		output += lines


		instruction_scopes[cur_scope] += (output)
		global_counters['instruction_counter'] += 1
	elif mode == 'cost':
		cost = 0
		total_cost += cost
	else:
		print(f'softmax(n,addr)')
		print([n,addr])

