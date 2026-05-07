use egg::{EGraph, Id};

use crate::ir::egraph::{TensorInfo, TensorOp};

pub fn enforce_alpha_injectivity(egraph: &mut EGraph<TensorOp, TensorInfo>) -> bool {
    let mut unions: Vec<(Id, Id)> = vec![];

    for class in egraph.classes() {
{{ALPHA_INJECTIVITY_DECLS}}

        for node in &class.nodes {
            match node {
{{ALPHA_INJECTIVITY_MATCH_ARMS}}
                _ => {}
            }
        }

{{ALPHA_INJECTIVITY_UNIONS}}
    }

    let mut changed = false;
    for (a, b) in unions {
        changed |= egraph.union(a, b);
    }

    if changed {
        egraph.rebuild();
    }

    changed
}

fn add_child_unions(unions: &mut Vec<(Id, Id)>, children: &[Id]) {
    let Some((&first, rest)) = children.split_first() else {
        return;
    };

    for &child in rest {
        if child != first {
            unions.push((first, child));
        }
    }
}
