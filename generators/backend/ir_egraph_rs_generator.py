import os
from typing import List
from dataclasses import dataclass
from .template_loader import get_backend_template_loader
from .alpha_utils import buffer_name_to_alpha_op, buffer_name_to_alpha_variant


@dataclass
class InstructionInfo:
    name: str
    arity: int
    has_metadata: bool


def get_rust_variant_name(instr_name: str) -> str:
    words = instr_name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in words)


def extract_instruction_info(instructions) -> List[InstructionInfo]:
    info_list = []

    for instruction in instructions:
        name = instruction.instruction
        arity = len(instruction.instr_inputs)
        has_metadata = len(instruction.comp_attr) > 0

        info_list.append(InstructionInfo(name, arity, has_metadata))

    return info_list


def generate_instruction_variants(instructions_info: List[InstructionInfo], templates):
    """Generate instruction variants using templates"""
    enum_variants = []
    num_children_arms = []
    set_metadata_arms = []
    children_arms = []
    children_mut_arms = []
    from_op_arms = []
    display_arms = []

    for info in instructions_info:
        variant_name = get_rust_variant_name(info.name)
        kebab_name = info.name.replace('_', '-')

        # Enum variant
        if info.has_metadata:
            signature = f"(String, [Id; {info.arity}])"
        else:
            signature = f"([Id; {info.arity}])"
        enum_variants.append(templates.render("egraph.rs.ISA_ENUM_VARIANTS.variant.txt",
                                             variant_name=variant_name,
                                             signature=signature))

        # Num children arm
        num_children_arms.append(templates.render("egraph.rs.ISA_NUM_CHILDREN_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name,
                                                  arity=str(info.arity)))

        # Set metadata arm (only for instructions with metadata)
        if info.has_metadata:
            set_metadata_arms.append(templates.render("egraph.rs.ISA_SET_METADATA_MATCH_ARMS.arm.txt",
                                                      variant_name=variant_name))

        # Children arms
        data_pattern = "(_, ids)" if info.has_metadata else "(ids)"
        children_arms.append(templates.render("egraph.rs.ISA_CHILDREN_MATCH_ARMS.arm.txt",
                                             variant_name=variant_name,
                                             data_pattern=data_pattern))
        children_mut_arms.append(templates.render("egraph.rs.ISA_CHILDREN_MUT_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name,
                                                  data_pattern=data_pattern))

        # From op arms
        if info.has_metadata:
            from_op_arms.append(templates.render("egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_with_metadata.txt",
                                                 kebab_name=kebab_name,
                                                 arity=str(info.arity),
                                                 variant_name=variant_name))
        else:
            from_op_arms.append(templates.render("egraph.rs.ISA_FROM_OP_MATCH_ARMS.arm_no_metadata.txt",
                                                 kebab_name=kebab_name,
                                                 arity=str(info.arity),
                                                 variant_name=variant_name))

        # Display arms
        if info.has_metadata:
            display_arms.append(templates.render("egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_with_metadata.txt",
                                                 variant_name=variant_name,
                                                 instruction_name=info.name))
        else:
            display_arms.append(templates.render("egraph.rs.ISA_DISPLAY_MATCH_ARMS.arm_no_metadata.txt",
                                                 variant_name=variant_name,
                                                 instruction_name=info.name))

    return {
        'enum_variants': enum_variants,
        'num_children_arms': num_children_arms,
        'set_metadata_arms': set_metadata_arms,
        'children_arms': children_arms,
        'children_mut_arms': children_mut_arms,
        'from_op_arms': from_op_arms,
        'display_arms': display_arms,
    }


def generate_alpha_variants(data_models, templates):
    """Generate alpha bridge operator variants from data models"""
    enum_variants = []
    num_children_arms = []
    children_arms = []
    children_mut_arms = []
    from_op_arms = []
    display_arms = []
    analysis_variants = []

    for data_model in data_models:
        variant_name = buffer_name_to_alpha_variant(data_model.var_name)
        op_name = buffer_name_to_alpha_op(data_model.var_name)

        enum_variants.append(templates.render("egraph.rs.ALPHA_ENUM_VARIANTS.variant.txt",
                                             variant_name=variant_name))
        num_children_arms.append(templates.render("egraph.rs.ALPHA_NUM_CHILDREN_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name))
        children_arms.append(templates.render("egraph.rs.ALPHA_CHILDREN_MATCH_ARMS.arm.txt",
                                             variant_name=variant_name))
        children_mut_arms.append(templates.render("egraph.rs.ALPHA_CHILDREN_MUT_MATCH_ARMS.arm.txt",
                                                  variant_name=variant_name))
        from_op_arms.append(templates.render("egraph.rs.ALPHA_FROM_OP_MATCH_ARMS.arm.txt",
                                             variant_name=variant_name,
                                             op_name=op_name))
        display_arms.append(templates.render("egraph.rs.ALPHA_DISPLAY_MATCH_ARMS.arm.txt",
                                             variant_name=variant_name,
                                             op_name=op_name))
        analysis_variants.append(f"TensorOp::{variant_name}(child)")

    if analysis_variants:
        analysis_arm = "            " + "\n            | ".join(analysis_variants) + " => egraph[*child].data.clone(),"
    else:
        analysis_arm = ""

    return {
        'enum_variants': enum_variants,
        'num_children_arms': num_children_arms,
        'children_arms': children_arms,
        'children_mut_arms': children_mut_arms,
        'from_op_arms': from_op_arms,
        'display_arms': display_arms,
        'analysis_arm': analysis_arm,
    }


def generate_egraph_rs_file(act_dest_dir, instructions, data_models):
    """Template egraph.rs from generic file"""
    templates = get_backend_template_loader()
    egraph_file = os.path.join(act_dest_dir, 'src', 'ir', 'egraph.rs')

    with open(egraph_file, 'r') as f:
        content = f.read()

    instructions_info = extract_instruction_info(instructions)
    instruction_parts = generate_instruction_variants(instructions_info, templates)
    alpha_parts = generate_alpha_variants(data_models, templates)

    replacements = {
        '{{ISA_ENUM_VARIANTS}}': '\n'.join(instruction_parts['enum_variants']),
        '{{ISA_NUM_CHILDREN_MATCH_ARMS}}': '\n'.join(instruction_parts['num_children_arms']),
        '{{ISA_SET_METADATA_MATCH_ARMS}}': '\n'.join(instruction_parts['set_metadata_arms']),
        '{{ISA_CHILDREN_MATCH_ARMS}}': '\n'.join(instruction_parts['children_arms']),
        '{{ISA_CHILDREN_MUT_MATCH_ARMS}}': '\n'.join(instruction_parts['children_mut_arms']),
        '{{ISA_FROM_OP_MATCH_ARMS}}': '\n'.join(instruction_parts['from_op_arms']),
        '{{ISA_DISPLAY_MATCH_ARMS}}': '\n'.join(instruction_parts['display_arms']),
        '{{ALPHA_ENUM_VARIANTS}}': '\n'.join(alpha_parts['enum_variants']),
        '{{ALPHA_NUM_CHILDREN_MATCH_ARMS}}': '\n'.join(alpha_parts['num_children_arms']),
        '{{ALPHA_CHILDREN_MATCH_ARMS}}': '\n'.join(alpha_parts['children_arms']),
        '{{ALPHA_CHILDREN_MUT_MATCH_ARMS}}': '\n'.join(alpha_parts['children_mut_arms']),
        '{{ALPHA_FROM_OP_MATCH_ARMS}}': '\n'.join(alpha_parts['from_op_arms']),
        '{{ALPHA_DISPLAY_MATCH_ARMS}}': '\n'.join(alpha_parts['display_arms']),
        '{{ALPHA_ANALYSIS_MATCH_ARM}}': alpha_parts['analysis_arm'],
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    with open(egraph_file, 'w') as f:
        f.write(content)
