import os
import re

from .alpha_utils import buffer_name_to_alpha_variant


def _bucket_name(var_name: str) -> str:
    return re.sub(r'\W+', '_', var_name).lower() + "_children"


def generate_alpha_rs_file(backend_gen_dir: str, data_models) -> None:
    """Fill alpha injectivity placeholders from data models."""
    alpha_file = os.path.join(backend_gen_dir, 'src', 'isel', 'rewrites', 'alpha.rs')
    with open(alpha_file, 'r') as f:
        content = f.read()

    decls = []
    match_arms = []
    unions = []

    for data_model in data_models:
        bucket = _bucket_name(data_model.var_name)
        variant = buffer_name_to_alpha_variant(data_model.var_name)
        decls.append(f"        let mut {bucket}: Vec<Id> = vec![];")
        match_arms.append(f"                TensorOp::{variant}(child) => {bucket}.push(egraph.find(*child)),")
        unions.append(f"        add_child_unions(&mut unions, &{bucket});")

    replacements = {
        '{{ALPHA_INJECTIVITY_DECLS}}': '\n'.join(decls),
        '{{ALPHA_INJECTIVITY_MATCH_ARMS}}': '\n'.join(match_arms),
        '{{ALPHA_INJECTIVITY_UNIONS}}': '\n'.join(unions),
    }

    for placeholder, replacement in replacements.items():
        content = content.replace(placeholder, replacement)

    with open(alpha_file, 'w') as f:
        f.write(content)
