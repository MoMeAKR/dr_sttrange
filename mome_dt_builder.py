import subprocess 
import os 
import glob 
import mcodeutils
import ast 
import astor 
import json


def _generate_node_function(node, utility_node):
    """
    Generates a Python function string for a single decision or terminal node.
    This helper prepares a list of statement dictionaries for mcodeutils.

    Fixes:
    - Avoids shadowing the function args list (['ctx']) by using `func_args`
      for the function signature and `util_args` for utility computations.
    - Ensures that local reads come from `ctx_to_use` (the possibly modified/copy ctx).
    """

    func_name = node['name']
    func_args = ['ctx']  # Function parameters must be only ['ctx'].
    stmts = []

    # Use a generic name for the local context copy for consistency.
    context_var = f"{func_name}_ctx"

    # Determine whether we should create a context copy.
    is_decision = node['node_type'] == 'decision'
    is_deterministic_terminal = node['node_type'] == 'terminal' and 'utility' in node
    has_modifications = bool(node.get('modifications'))
    needs_copy = has_modifications or is_decision or is_deterministic_terminal
    ctx_to_use = context_var if needs_copy else 'ctx'

    # Add a docstring using the node's description.
    stmts.append({'type': 'raw', 'value': f'"""{node["desc"]}"""'})

    # Create a context copy and apply modifications if required.
    if needs_copy:
        stmts.append({'type': 'assign', 'var': ctx_to_use, 'value': 'ctx.copy()'})
        for key, value in node.get('modifications', {}).items():
            # Use raw type for dict assignment, which is parsed as a statement.
            stmts.append({'type': 'raw', 'value': f"{ctx_to_use}['{key}'] = {repr(value)}"})

    desc = node['desc']

    # --- Return Statement Logic ---
    if node['node_type'] == 'decision':
        children = node['children']
        # Handle single, deterministic child branch (prob=1.0)
        if len(children) == 1 and children[0]['prob'] == 1.0:
            lambda_expr = f"lambda _: {children[0]['child']}({ctx_to_use})"
            return_expr = f"('{desc}', None, {lambda_expr})"
        # Handle multiple probabilistic child branches
        else:
            lambdas = [f"({c['prob']}, lambda _: {c['child']}({ctx_to_use}))" for c in children]
            return_expr = f"('{desc}', None, [{', '.join(lambdas)}])"
        stmts.append({'type': 'return', 'var': return_expr})

    elif node['node_type'] == 'terminal':
        # Handle stochastic terminal nodes (with probabilistic outcomes)
        if 'children' in node and node['children']:
            # Extract context variables needed for utility calculations.
            ctx_vars = {
                v[v.find("'") + 1:v.rfind("'")]
                for c in node['children']
                for v in c.get('utility', {}).values()
                if isinstance(v, str) and v.startswith("ctx[")
            }
            # Bind the referenced context values into local variables.
            for var in sorted(list(ctx_vars)):
                stmts.append({'type': 'assign', 'var': var, 'value': f"{ctx_to_use}['{var}']"})

            # Build the list of outcome tuples (prob, utility_call, outcome_desc)
            outcomes = []
            for child in node['children']:
                util_args = [
                    child['utility'][uv].replace(f"ctx['{uv}']", uv) if isinstance(child['utility'][uv], str)
                    else str(child['utility'][uv])
                    for uv in utility_node['variables']
                ]
                utility_call = f"{utility_node['name']}({', '.join(util_args)})"
                outcomes.append(f"({child['prob']}, {utility_call}, '{child['outcome_desc']}')")

            return_expr = f"('{desc}', [{', '.join(outcomes)}], None)"
            stmts.append({'type': 'return', 'var': return_expr})

        # Handle deterministic terminal nodes (with a single utility payoff)
        elif 'utility' in node:
            util_args = [
                str(node['utility'][uv]).replace('ctx', ctx_to_use)
                for uv in utility_node['variables']
            ]
            utility_call = f"{utility_node['name']}({', '.join(util_args)})"
            return_expr = f"('{desc}', {utility_call}, None)"
            stmts.append({'type': 'return', 'var': return_expr})

    # Important: pass func_args (['ctx']) so function signatures stay correct.
    return mcodeutils.simple_function_from_dicts(
        func_name=func_name,
        args=func_args,
        stmts=stmts
    )






# def exp():
#     results = mome_dt_commons.enumerate_paths(root, {})
#     expected_utility = mome_dt_commons.expected_utility(root, {})
#     agg_results = {'tree_paths': results, 'expected_utility': expected_utility}
#     save_path = '/home/mehdimounsif/Codes/my_libs/mome_dt/tmp_results/dt_results.json'
#     os.makedirs('/home/mehdimounsif/Codes/my_libs/mome_dt/tmp_results', exist_ok=True)
#     with open(save_path, 'w') as f:
#         json.dump(agg_results, f, indent=4)
#     return save_path

# if __name__ == '__main__':
#     exp()

def default_exp(params): 

    
    current_save_path = params['exp_management'].get("results_file", os.path.join(os.path.dirname(__file__), "tmp_results", "default_dt_results.json"))
    # my_val = ast.unparse(mcodeutils.get_right_side(astor.to_source(mcodeutils.simple_dict_assign_ast("tmp", ["tree_paths"], ['results'])), "tmp"))
    # input(my_val)
    stmts = [
        {"type": "call", "assign_var": "results", "func": "mome_dt_commons.enumerate_paths", "args": ["root", "dict()"]}, 
        {"type": "call", "assign_var": "expected_utility", "func": "mome_dt_commons.expected_utility", "args": ["root", "dict()"]},
        {"type": "raw", "value": mcodeutils.simple_dict_assign_ast("agg_results", ["tree_paths", "expected_utility", "plots"], ["results", "expected_utility", []])}, 
        {"type": "raw", "value": mcodeutils.simple_dict_assign_ast("plot_data", ["initial_run"], [ast.unparse(mcodeutils.get_right_side(astor.to_source(mcodeutils.simple_dict_assign_ast("tmp", ["tree_paths"], ['results'])), "tmp"))])},
        {"type": "call", "func": "mome_dt_commons.plot_multiple_outcome_probabilities", "args": ['plot_data', repr(os.path.join(os.path.dirname(__file__), "tmp_outcomes.jpg"))]},
        {"type": "call", "func": "mome_dt_commons.plot_cumulative_risk", "args": ['plot_data', repr(os.path.join(os.path.dirname(__file__), "tmp_cumulative_risk.jpg"))]},
        {"type": "append", "iterable" : "agg_results['plots']", "value": repr(os.path.join(os.path.dirname(__file__), "tmp_outcomes.jpg"))}, 
        {"type": "append", "iterable" : "agg_results['plots']", "value": repr(os.path.join(os.path.dirname(__file__), "tmp_cumulative_risk.jpg"))}, 
        {"type": "assign", "var": "save_path", "value": repr(current_save_path)}, 
        {"type": "call", "func": "os.makedirs", "args": [astor.to_source(mcodeutils.stmt_dict_to_ast({"type": "call", "func": "os.path.dirname", "args": [repr(current_save_path)]}))], "named_args": {"exist_ok": True}},
        {"type": "save_json", "where": "save_path", "save_var": "agg_results"},
        {"type": "return", "var": "save_path"}
    ]

    func_code = mcodeutils.simple_function_from_dicts(
        func_name = params.get("func_name", "exp"), 
        args = [],
        stmts = stmts
    )


    return func_code


def build_result(data):
    """
    Generates a Python script from a dictionary describing a decision tree structure.

    This function processes a list of nodes, identifies a special 'utility' node,
    and then generates a corresponding Python function for each node, using
    helper functions from the `mcodeutils` module to build the code from ASTs.

    Args:
        data: A dictionary containing a 'nodes' key with a list of node definitions.

    Returns:
        A string containing the complete, runnable Python script.
    """
    all_funcs_code = []

    # Find the utility node, which defines the final payoff calculation.
    utility_params = data.get("utility", {})
    if len(utility_params) > 0: 

        # 1. Generate the utility function first.
        utility_stmts = [
            {'type': 'raw', 'value': f'"""{utility_params["desc"]}"""'},
            {'type': 'assign', 'var': 'u', 'value': utility_params['operation']},
            {'type': 'return', 'var': 'u'}
        ]
        utility_func_code = mcodeutils.simple_function_from_dicts(
            func_name=utility_params['name'],
            args=utility_params['variables'],
            stmts=utility_stmts
        )
        all_funcs_code.append(utility_func_code)

    # 2. Add a separator for readability.
    all_funcs_code.append('\n# ---------- Nodes ----------')

    # 3. Generate functions for all decision and terminal nodes.
    # other_nodes = [n for n in data['nodes'] if n['node_type'] != 'utility']

    for node in data['nodes']:
        node_func_code = _generate_node_function(node, utility_params)
        all_funcs_code.append(node_func_code)

    # Join all generated function strings with double newlines for spacing.
    func_code =  "\n\n".join(all_funcs_code)
    return func_code

def prepare_main_default(params): 

    main_block = astor.to_source(mcodeutils.create_main_block([mcodeutils.stmt_dict_to_ast({"type": "call", "func": params.get("func_name", "exp")})]))
    return main_block


def prepare_dt(params): 

    all_functions = build_result(params)
    
    exp_type = params.get('exp_management', {}).get("exp_type", "default").strip()
    
    if exp_type == "default": 
        entry_point = default_exp(params)    
    else:
        print('Missing experience type: {}. Defaulting exp function ')
        entry_point = default_exp(params)    

    
    imports = ["import json", "import os", "import mome_dt_commons", "import sys", "import numpy as np"]    
    
    main_block = prepare_main_default(params)
    
    
    final_code = mcodeutils.refactor_imports("\n\n".join(["\n".join(imports), all_functions, entry_point, main_block]))
    return final_code


if __name__ == '__main__':
    files = glob.glob(os.path.join(os.path.dirname(__file__), "test_material", "*.json"))
    for file in files: 
        
        source_data = json.load(open(file))

        generated_code = build_result(source_data)

        generated_code += """

ctx= {}


print("All paths:")
for p, u, path in mome_dt_commons.enumerate_paths(root, ctx):
    print("Prob: {} -> Utility {} via {}".format(p,u,path))
print("Expected utility:", mome_dt_commons.expected_utility(root, ctx))

"""
        generated_code = "\n".join(["import numpy as np", "import mome_dt_commons"]) + "\n\n{}".format(generated_code)
        # input(generated_code)
        generated_code = mcodeutils.refactor_imports(generated_code)
        
        with open(os.path.join(os.path.dirname(file), "result_code.py"), "w") as f: 
            f.write(generated_code)
        result = subprocess.run(['python', os.path.join(os.path.dirname(file), "result_code.py")], capture_output= True, text = True)
        print('File: {}'.format(file))
        print(result.stdout)
        print('\n\n\n')




