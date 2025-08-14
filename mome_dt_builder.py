import momeutils 
import subprocess 
import os 
import glob 
import mcodeutils
import ast 
import astor 
import json


def generate_node_function(node, utility_node):
    """
    Dispatches node function generation to the appropriate handler
    based on node_type: 'utility', 'decision', or 'terminal'.
    """
    # node_type = node['node_type']
    
    node_type = node.get("node_type", "terminal" if len(node['children']) == 1 else "decision")

    # if node_type == 'root':
    #     # return _generate_utility_node_function(node)
    #     return _generate_root_node_function(node,utility_node)
    # elif node_type == 'decision':
    if node_type == 'decision':
        return _generate_decision_node_function(node)
    elif node_type == 'terminal':
        return _generate_terminal_node_function(node, utility_node)
    else:
        raise ValueError(f"Unknown node_type: {node_type}")


def _generate_root_node_function(node,utility_node): 
    """
    TARGET: 
    def root(ctx):
    "Select firm"
    root_ctx = ctx.copy()
    root_ctx['money'] = 0
    root_ctx['transferable_expertise'] = 0
    root_ctx['industry_credibility'] = 0
    return ('Select firm', None, [(0.5, lambda _: european_project(root_ctx)), (0.5, lambda _: client_work(root_ctx))])

    """
    # print(node)
    # print('\n')
    # print(utility_node)
    stmts = []
    stmts.append({"type": "comment", "text": node['desc']})
    stmts.append({"type": "assign", "var": "{}_ctx".format(node['node_id']), "value": "ctx.copy()"})
    for uv in utility_node['variables']: 
        stmts.append({"type": "assign", "var": "{}_ctx['{}']".format(node['node_id'], uv), "value": int("0")})
    
    stmts.append({"type": "assign", "var": "children", "value": []})
    for child in node['children']: 
        # lamb_func = compile(mcodeutils.simple_lambda_func(['_'], "{}(ctx)".format(child['child'])), filenode_id = "<ast>", mode = "eval")
        # input(lamb_func)
        # input(astor.to_source(lamb_func))
        stmts.append({"type": "append", "iterable": "children", "value": (0.5, ast.parse("lambda _ : {}({})".format(child['child'], '{}_ctx'.format(node['node_id']))))})

    stmts.append({"type": "assign", "var": "func_results", "value": "('{}', {}, children)".format(node['desc'], None)})
    stmts.append({"type": "return", "var": "func_results"})
    func = mcodeutils.simple_function_from_dicts(func_name = "root", 
                                                 args = ['ctx'], 
                                                 stmts = stmts)
    return func 


def _generate_decision_node_function(node):
    """
    Generates a Python function string for a decision node.
    Handles deterministic and probabilistic child branches.

    target 
    def european_project_major_role_outcome(ctx):
        "Major role in a European project leads to significant expertise and credibility gains, plus financial reward."
        european_project_major_role_outcome_ctx = ctx.copy()
        european_project_major_role_outcome_ctx['money'] = 3
        european_project_major_role_outcome_ctx['transferable_expertise'] = 5
        european_project_major_role_outcome_ctx['industry_credibility'] = 5

        return ('Major role in a European project leads to significant expertise and credibility gains, plus financial reward.', [(1, utility(european_project_major_role_outcome_ctx['money'], european_project_major_role_outcome_ctx['transferable_expertise'], european_project_major_role_outcome_ctx['industry_credibility']), 'Successful leadership or major contribution in a European project yields strong professional and financial benefits.')], None)
    """
    func_name = node['node_id']
    func_args = ['ctx']
    stmts = []
    context_var = f"{func_name}_ctx"
    desc = node['desc']
    children = node['children']

    # Add docstring
    stmts.append({'type': 'raw', 'value': f'"""{desc}"""'})

    # Always copy context for decision nodes
    stmts.append({'type': 'assign', 'var': context_var, 'value': 'ctx.copy()'})
    for k in node["modifications"].keys(): 
        stmts.append({'type': 'assign', 'var': "{}['{}']".format(context_var, k), 'value': node['modifications'][k]})
    ctx_to_use = context_var

    # Handle deterministic (single child, prob=1.0) and probabilistic branches
    if len(children) == 1 and children[0]['prob'] == 1.0:
        lambda_expr = f"lambda _: {children[0]['child']}({ctx_to_use})"
        return_expr = f"('{desc}', None, {lambda_expr})"
    else:
        lambdas = [f"({c['prob']}, lambda _: {c['child']}({ctx_to_use}))" for c in children]
        return_expr = f"('{desc}', None, [{', '.join(lambdas)}])"
    stmts.append({'type': 'return', 'var': return_expr})


    func = mcodeutils.simple_function_from_dicts(
        func_name=func_name,
        args=func_args,
        stmts=stmts
    )

    return func

def _generate_terminal_node_function(node, utility_node):
    """
    Generates a Python function string for a terminal node.
    Handles both deterministic and stochastic (probabilistic) terminal nodes.
    """
    func_name = node['node_id']
    func_args = ['ctx']
    stmts = []
    context_var = f"{func_name}_ctx"
    desc = node['desc']

    # Determine if context copy is needed
    has_modifications = bool(node.get('modifications'))
    is_deterministic_terminal = 'utility' in node
    needs_copy = has_modifications or is_deterministic_terminal
    ctx_to_use = context_var if needs_copy else 'ctx'

    # Add docstring
    stmts.append({'type': 'comment', 'text': desc})

    # Copy context and apply modifications if needed
    if needs_copy:
        stmts.append({'type': 'assign', 'var': ctx_to_use, 'value': 'ctx.copy()'})
        for key, value in node.get('modifications', {}).items():
            stmts.append({'type': 'raw', 'value': f"{ctx_to_use}['{key}'] = {repr(value)}"})

    # Stochastic terminal: multiple children with probabilities
    if 'children' in node and node['children']:
        # Collect context variables referenced in utility expressions
        ctx_vars = {
            v[v.find("'") + 1:v.rfind("'")]
            for c in node['children']
            for v in c.get('utility', {}).values()
            if isinstance(v, str) and v.startswith("ctx[")
        }
        for var in sorted(ctx_vars):
            stmts.append({'type': 'assign', 'var': var, 'value': f"{ctx_to_use}['{var}']"})

        # Build outcome tuples: (prob, utility_call, outcome_desc)
        outcomes = []
        for child in node['children']:
            util_args = ["{}['{}']".format(ctx_to_use, k) for k in utility_node['variables']]
            utility_call = f"{utility_node['name']}({', '.join(util_args)})"
            outcomes.append(f"({child['prob']}, {utility_call}, '{child['outcome_desc']}')")
        return_expr = f"('{desc}', [{', '.join(outcomes)}], None)"
        stmts.append({'type': 'return', 'var': return_expr})

    # Deterministic terminal: single utility payoff
    elif 'utility' in node:
        util_args = [
            str(node['utility'][uv]).replace('ctx', ctx_to_use)
            for uv in utility_node['variables']
        ]
        utility_call = f"{utility_node['name']}({', '.join(util_args)})"
        return_expr = f"('{desc}', {utility_call}, None)"
        stmts.append({'type': 'return', 'var': return_expr})

    return mcodeutils.simple_function_from_dicts(
        func_name=func_name,
        args=func_args,
        stmts=stmts
    )

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

    all_funcs_code.append(_generate_root_node_function(data['root'], utility_params))

    for i, node in enumerate(data['nodes']):

        print('processing node {}'.format(node['node_id']))
        node_func_code = generate_node_function(node, utility_params)
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




