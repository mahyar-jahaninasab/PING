import re
import ast
import json
from pathlib import Path
import os




class ODEParser:


    def __init__(self):
        self.config = {}
        self.variables = {}
        self.matrices = {}
        self.equations = {}
        self.derivative_spec = {}
        self.derivative_constraints = []
        self.nonlinear_terms = {}
        self.nonlinear_terms2 = {}
        self.equations2 = {}
        self.state_constraints = []
        self.ode_equation = []
        self.action_constraints = []  


        self.supported_operations = {
            'sin': 'torch.sin', 'cos': 'torch.cos', 'tan': 'torch.tan',
            'asin': 'torch.asin', 'acos': 'torch.acos', 'atan': 'torch.atan',
            'sinh': 'torch.sinh', 'cosh': 'torch.cosh', 'tanh': 'torch.tanh',
            'exp': 'torch.exp', 'log': 'torch.log', 'log10': 'torch.log10',
            'log2': 'torch.log2', 'sqrt': 'torch.sqrt', 'pow': 'torch.pow',
            'square': 'torch.square', 'abs': 'torch.abs', 'ceil': 'torch.ceil',
            'floor': 'torch.floor', 'round': 'torch.round',
            'sigmoid': 'torch.sigmoid', 'relu': 'torch.relu',
            'clamp': 'torch.clamp', 'sign': 'torch.sign',
        }


        self.supported_binary_ops = {
            ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y, ast.Div: lambda x, y: x / y,
            ast.Pow: lambda x, y: x ** y, ast.FloorDiv: lambda x, y: x // y,
            ast.Mod: lambda x, y: x % y,
        }


        self.supported_unary_ops = {
            ast.UAdd: lambda x: +x, ast.USub: lambda x: -x,
        }


    def _is_action_constraint(self, line):
        """Check if line contains action constraint like: action_0 > 0"""
        constraint_ops = ['>=', '<=', '>', '<', '==']
        has_action = 'action_' in line
        has_operator = any(op in line for op in constraint_ops)
        has_equals = '=' in line
        return has_action and has_operator and (not has_equals or any(op in line for op in constraint_ops[:-1]))


    def _parse_action_constraint(self, line):
        constraint_op = None
        for op in ['>=', '<=', '>', '<', '==']:
            if op in line:
                constraint_op = op
                left, right = line.split(op, 1)
                left = left.strip()
                right = right.strip()
                break


        if not constraint_op:
            return None


        action_match = re.search(r'action_\d+', left)
        if not action_match:
            return None


        action_name = action_match.group(0)
        try:
            bound_value = float(right)
        except:
            bound_value = right 
        return {
            'action': action_name,
            'operator': constraint_op,
            'bound': bound_value,
            'original': line.strip()
        }


    def _extract_variables_from_equation(self, equation_str):
        """Extract all variable names from an equation string"""
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        matches = re.findall(pattern, equation_str)


        keywords = {'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while'}
        variables = [m for m in matches if m not in keywords]


        return list(set(variables))


    def parse_ode_text(self, text):
        """Enhanced parser that handles ODE equations and action constraints"""


        lines = text.strip().split('\n')
        current_section = None
        current_subsection = None
        constraint_buffer = ""


        self.config = {
            '#Domain': {},
            '#Constraints': [],
            '#Reward': {
                'Initial_Condition': [],
                'End_Point_Condition': [],
                'State_Condition': [],
                'optimization': {},
                'rate': None,
                'specific_conditions': []
            },
            '#ODE_System': {
                'derivative_spec': {},
                'matrices': {},
                'equations': {}
            },
            '#Derivative_Value_Constraints': [],
            '#NonLinear_Terms': {},
            '#NonLinear_Terms2': {},
            '#Equations': {},
            '#Action_Constraints': []  
        }


        self.ode_equation = []
        self.action_constraints = []
        self.nonlinear_terms = {}
        self.nonlinear_terms2 = {}
        self.equations2 = {}
        ode_mode = False
        action_constraint_mode = False


        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                if line.startswith('#'):
                    if constraint_buffer and current_section == '#Constraints':
                        self._process_constraint_group(constraint_buffer)
                        constraint_buffer = ""


                    if 'ODE system' in line or 'ODE System' in line:
                        current_section = '#ODE_System'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Define NonLinear Terms 2' in line or 'NonLinear Terms 2' in line or 'Nonlinear Terms 2' in line:
                        current_section = '#NonLinear_Terms2'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Define NonLinear Terms' in line or 'NonLinear Terms' in line or 'Nonlinear Terms' in line:
                        current_section = '#NonLinear_Terms'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif line == '#ODE' or 'ODE' in line:
                        current_section = '#ODE'
                        ode_mode = True
                        action_constraint_mode = False
                    elif 'action condition' in line.lower():
                        current_section = '#Action_Constraints'
                        ode_mode = False
                        action_constraint_mode = True
                    elif line.startswith('##'):
                        if 'Initial Condition' in line:
                            current_subsection = 'Initial_Condition'
                        elif 'End Point Condition' in line or 'Endpoint Condition' in line:
                            current_subsection = 'End_Point_Condition'
                        elif 'state condition' in line or 'State Condition' in line or 'State condition' in line:
                            current_subsection = 'State_Condition'
                    elif 'Domain' in line:
                        current_section = '#Domain'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Constraint' in line and 'Derivative' not in line:
                        current_section = '#Constraints'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Reward' in line:
                        current_section = '#Reward'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Derivative' in line and 'Value' in line and 'Constraint' in line:
                        current_section = '#Derivative_Value_Constraints'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                    elif 'Equations' in line or 'Euqations' in line:
                        current_section = '#Equations'
                        current_subsection = None
                        ode_mode = False
                        action_constraint_mode = False
                continue


            if ode_mode and '=' in line:
                self.ode_equation.append(line.strip())
                variables = self._extract_variables_from_equation(line)
                for var in variables:
                    if var not in self.variables:
                        self.variables[var] = {}
                continue


            if action_constraint_mode or self._is_action_constraint(line):
                constraint = self._parse_action_constraint(line)
                if constraint:
                    self.action_constraints.append(constraint)
                    self.config['#Action_Constraints'].append(constraint)
                continue


            if current_section == '#Domain':
                self._parse_domain(line)


            elif current_section == '#Constraints':
                constraint_buffer += " " + line
                if not line.strip().endswith('or'):
                    self._process_constraint_group(constraint_buffer)
                    constraint_buffer = ""


            elif current_section == '#Reward':
                if current_subsection == 'Initial_Condition':
                    if 'state_' in line and '(' in line and 't' in line and '=' in line:
                        self._parse_initial_condition(line)
                    elif not line.startswith('##'):
                        current_subsection = None
                        self._parse_reward_general(line)


                elif current_subsection == 'End_Point_Condition':
                    if 'state_' in line and '(' in line and 't' in line and '=' in line:
                        self._parse_endpoint_condition(line)
                    elif not line.startswith('##'):
                        current_subsection = None
                        self._parse_reward_general(line)


                elif current_subsection == 'State_Condition':
                    is_specific_condition = bool(re.match(r'state_\d+\s*\([^)]*\)\s*=', line))
                    if not is_specific_condition and ('&' in line or '<' in line or '>' in line or 'state_' in line):
                        self._parse_reward_state_condition(line)
                    elif is_specific_condition:
                        self._parse_specific_state_condition(line)
                    elif not line.startswith('##'):
                        current_subsection = None
                        self._parse_reward_general(line)
                else:
                    self._parse_reward_general(line)


            elif current_section == '#ODE_System':
                if self._is_derivative_spec(line):
                    self._parse_derivative_spec(line)
                elif '[[' in line and ']]' in line:
                    self._parse_matrix(line)
                elif '=' in line and not any(op in line for op in ['<', '>', '!=']):
                    self._parse_equation(line)


            elif current_section == '#Derivative_Value_Constraints':
                if self._is_derivative_constraint(line):
                    constraint = self._parse_derivative_constraint(line)
                    if constraint:
                        self.derivative_constraints.append(constraint)
                        self.config['#Derivative_Value_Constraints'].append(constraint)


            elif current_section == '#NonLinear_Terms':
                if '=' in line and not any(op in line for op in ['<', '>', '!=']):
                    self._parse_nonlinear_term(line, 'nonlinear_terms')
            
            elif current_section == '#NonLinear_Terms2':
                if '=' in line and not any(op in line for op in ['<', '>', '!=']):
                    self._parse_nonlinear_term(line, 'nonlinear_terms2')


            elif current_section == '#Equations':
                if '=' in line and not any(op in line for op in ['<', '>', '!=']):
                    self._parse_nonlinear_term(line, 'equations2')


        if constraint_buffer and current_section == '#Constraints':
            self._process_constraint_group(constraint_buffer)


        self.config['#Constraints'] = self.state_constraints
        self.config['#ODE_System']['derivative_spec'] = self.derivative_spec
        self.config['#ODE_System']['matrices'] = self.matrices
        self.config['#ODE_System']['equations'] = self.equations
        self.config['#NonLinear_Terms'] = self.nonlinear_terms
        self.config['#NonLinear_Terms2'] = self.nonlinear_terms2
        self.config['#Equations'] = self.equations2
        self.config['#ODE_System']['calculate_dervative'] = self.get_calculate_dervative_config()
        self._normalize_state_variables()


        return self


    def get_action_info(self):
        """
        Extract action-related information from parsed equations.
        Returns dict with actions, equations using them, and constraints.
        """
        actions = set()
        for eq in self.ode_equation:
            action_matches = re.findall(r'action_\d+', eq)
            actions.update(action_matches)


        return {
            'actions': sorted(list(actions)),
            'num_actions': len(actions),
            'equations': self.ode_equation,
            'constraints': self.action_constraints
        }


    def _normalize_state_variables(self):
        num_states = self.config['#Domain'].get('states', 0)
        if num_states == 0:
            return


        expected_states = [f'state_{i}' for i in range(num_states)]


        for constraint in self.config['#Constraints']:
            if 'states' in constraint:
                for state_name in expected_states:
                    if state_name not in constraint['states']:
                        constraint['states'][state_name] = {}


        for condition in self.config['#Reward']['State_Condition']:
            if 'states' in condition:
                for state_name in expected_states:
                    if state_name not in condition['states']:
                        condition['states'][state_name] = {}


    def _parse_domain(self, line):
        if 'states' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['states'] = int(value.strip())
        elif 'layers' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['layers'] = int(value.strip())
        elif 'nodes' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['nodes'] = int(value.strip())
        elif 'batch_size' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['batch_size'] = int(value.strip())
        elif 'numbers' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['numbers'] = int(value.strip())
        elif 'num_points' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['num_points'] = int(value.strip())
        elif 'actions' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['actions'] = int(value.strip())
        elif 'upper_bound_time' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['upper_bound_time'] = float(value.strip())
        elif 'rate' in line.lower() and '=' in line:
            _, value = line.split('=')
            self.config['#Domain']['rate'] = float(value.strip())            
        elif '<' in line and 'time' in line.lower():
            self.config['#Domain']['time_constraint'] = line.strip()
            time_bounds = self._parse_time_bounds(line)
            if time_bounds:
                self.config['#Domain']['time_bounds'] = time_bounds


    def _parse_time_bounds(self, line):
        result = {}
        pattern = r'([-\d.]+)\s*([<>]=?)\s*time\s*([<>]=?)\s*([-\d.]+)'
        match = re.search(pattern, line)


        if match:
            value1 = float(match.group(1))
            op1 = match.group(2)
            op2 = match.group(3)
            value2 = float(match.group(4))


            result['lower'] = (value1, op1)
            result['upper'] = (value2, op2)


        return result


    def _parse_initial_condition(self, line):
        pattern = r'(state_\d+)\s*\(\s*t\s*=\s*([-\d.]+)\s*\)\s*=\s*([-\d.]+)'
        matches = re.findall(pattern, line)


        for match in matches:
            self.config['#Reward']['Initial_Condition'].append(float(match[2]))


    def _parse_endpoint_condition(self, line):
        pattern = r'(state_\d+)\s*\(\s*t\s*=\s*([-\w.]+)\s*\)\s*=\s*([-\d.]+)'
        matches = re.findall(pattern, line)


        for match in matches:
            state_name = match[0]
            time_value = match[1]
            target_value = float(match[2])


            is_endpoint = False
            time_pointer = 0


            try:
                time_numeric = float(time_value)
                if time_numeric == -1:
                    is_endpoint = True
                    time_pointer = -1
                else:
                    time_pointer = time_numeric
            except ValueError:
                is_endpoint = False
                time_pointer = 0


            self.config['#Reward']['End_Point_Condition'].append({
                'state': state_name,
                'time_value': time_value,
                'target_value': target_value,
                'is_endpoint': is_endpoint,
                'time_pointer': time_pointer
            })


    def _parse_reward_state_condition(self, line):
        if '&' in line or '<' in line or '>' in line:
            parsed = self._parse_state_constraint(line)
            if parsed:
                self.config['#Reward']['State_Condition'].append(parsed)


    def _parse_reward_general(self, line):
        if line.startswith('Min') or line.startswith('Max'):
            obj_match = re.match(r'(Min|Max)\s*\(\s*(\w+)\s*\)', line)
            if obj_match:
                self.config['#Reward']['optimization'] = {
                    'type': obj_match.group(1),
                    'variable': obj_match.group(2)
                }


        elif line.startswith('rate') and '=' in line:
            _, value = line.split('=')
            self.config['#Reward']['rate'] = float(value.strip())


        elif 'state_' in line and '(' in line and ')' in line and '=' in line:
            self._parse_specific_state_condition(line)


    def _parse_specific_state_condition(self, line):
        pattern = r'(state_\d+)\s*\(\s*([^)]+)\s*\)\s*=\s*(.+)'
        match = re.search(pattern, line)


        if match:
            state_name = match.group(1)
            time_condition = match.group(2).strip()
            value = match.group(3).strip()


            time_bounds = {}
            time_pattern = r'([-\d.]+)\s*([<>]=?)\s*t\s*([<>]=?)\s*([-\d.]+)'
            time_match = re.search(time_pattern, time_condition)


            if time_match:
                time_bounds = {
                    'lower': {'value': float(time_match.group(1)), 'operator': time_match.group(2)},
                    'upper': {'value': float(time_match.group(4)), 'operator': time_match.group(3)}
                }


            self.config['#Reward']['specific_conditions'].append({
                'state': state_name,
                'time_condition': time_condition,
                'time_bounds': time_bounds,
                'value': value,
            })


    def _parse_matrix(self, line):
        var_name, matrix_str = line.split('=')
        var_name = var_name.strip()
        matrix_data = eval(matrix_str.strip())
        self.matrices[var_name] = matrix_data


    def _parse_equation(self, line):
        var_name, expr_str = line.split('=', 1)
        var_name = var_name.strip()
        expr_str = expr_str.strip()
        self.equations[var_name] = expr_str


    def _process_constraint_group(self, text):
        or_groups = [g.strip() for g in re.split(r'\s+or\s+', text) if g.strip()]
        for group in or_groups:
            constraint = self._parse_state_constraint(group)
            if constraint:
                self.state_constraints.append(constraint)


    def _parse_state_constraint(self, line):
        conditions = [c.strip() for c in line.split('&')]
        constraint = {'time': {}, 'states': {}}


        for condition in conditions:
            condition = condition.strip()
            if condition.startswith('(') and condition.endswith(')'):
                condition = condition[1:-1].strip()


            if self._is_time_constraint(condition):
                time_constraint = self._parse_time_constraint(condition)
                if time_constraint:
                    constraint['time'].update(time_constraint)


            elif 'state_' in condition:
                state_constraints = self._parse_state_bounds(condition)
                for state_name, bounds in state_constraints.items():
                    if state_name not in constraint['states']:
                        constraint['states'][state_name] = {}
                    constraint['states'][state_name].update(bounds)


        return constraint


    def _is_time_constraint(self, condition):
        return bool(re.search(r'\bt\b', condition))


    def _parse_time_constraint(self, condition):
        result = {}


        pattern_compound = r'([-\d.]+)\s*([<>]=?)\s*t\s*([<>]=?)\s*([-\d.]+)'
        match_compound = re.search(pattern_compound, condition)


        if match_compound:
            value1 = float(match_compound.group(1))
            op1 = match_compound.group(2)
            op2 = match_compound.group(3)
            value2 = float(match_compound.group(4))


            if '<' in op1:
                result['lower'] = (value1, op1)
            else:
                result['upper'] = (value1, op1)


            if '<' in op2:
                result['upper'] = (value2, op2)
            else:
                result['lower'] = (value2, op2)


            return result


        pattern1 = r'([-\d.]+)\s*([<>]=?)\s*t\b'
        match1 = re.search(pattern1, condition)
        if match1:
            value = float(match1.group(1))
            operator = match1.group(2)


            if '<' in operator:
                result['lower'] = (value, operator)
            else:
                result['upper'] = (value, operator)


            return result


        pattern2 = r't\b\s*([<>]=?)\s*([-\d.]+)'
        match2 = re.search(pattern2, condition)


        if match2:
            operator = match2.group(1)
            value = float(match2.group(2))
            if '>' in operator:
                result['lower'] = (value, operator)
            else:
                result['upper'] = (value, operator)


            return result


        return result


    def _parse_state_bounds(self, condition):
        results = {}


        pattern_compound = r'([-\d.]+)\s*([<>]=?)\s*(state_\d+)\s*([<>]=?)\s*([-\d.]+)'
        match_compound = re.search(pattern_compound, condition)


        if match_compound:
            value1 = float(match_compound.group(1))
            op1 = match_compound.group(2)
            state_name = match_compound.group(3)
            op2 = match_compound.group(4)
            value2 = float(match_compound.group(5))


            results[state_name] = {}


            if '<' in op1:
                results[state_name]['lower'] = (value1, op1)
            else:
                results[state_name]['upper'] = (value1, op1)


            if '<' in op2:
                results[state_name]['upper'] = (value2, op2)
            else:
                results[state_name]['lower'] = (value2, op2)


            return results


        pattern_simple1 = r'(state_\d+)\s*([<>]=?)\s*([-\d.]+)'
        match_simple1 = re.search(pattern_simple1, condition)


        if match_simple1:
            state_name = match_simple1.group(1)
            operator = match_simple1.group(2)
            value = float(match_simple1.group(3))


            results[state_name] = {}


            if '>' in operator:
                results[state_name]['lower'] = (value, operator)
            else:
                results[state_name]['upper'] = (value, operator)


            return results


        pattern_simple2 = r'([-\d.]+)\s*([<>]=?)\s*(state_\d+)'
        match_simple2 = re.search(pattern_simple2, condition)


        if match_simple2:
            value = float(match_simple2.group(1))
            operator = match_simple2.group(2)
            state_name = match_simple2.group(3)
            results[state_name] = {}
            if '<' in operator:
                results[state_name]['lower'] = (value, operator)
            else:
                results[state_name]['upper'] = (value, operator)
            return results


        return results


    def _is_derivative_spec(self, line):
        return ('d' in line and '/' in line and 'dt' in line and 'state_' in line 
                and not any(op in line for op in ['=', '<', '>', '<=', '>=']))


    def _is_derivative_constraint(self, line):
        return ('d' in line and '/' in line and 'dt' in line and 'state_' in line 
                and any(op in line for op in ['=', '<', '>', '<=', '>=']))


    def _parse_derivative_spec(self, line):
        pattern = r'd\s+state_(\d+)\s*/\s*dt\s*(?:\((\d+)\))?'
        matches = re.findall(pattern, line)
        for match in matches:
            state_idx = int(match[0])
            order = int(match[1]) if match[1] else 1
            if state_idx not in self.derivative_spec:
                self.derivative_spec[state_idx] = []
            if order not in self.derivative_spec[state_idx]:
                self.derivative_spec[state_idx].append(order)


        for state_idx in self.derivative_spec:
            self.derivative_spec[state_idx].sort()


    def _parse_derivative_constraint(self, line):
        pattern = r'd\s+(state_\d+)\s*/\s*dt\s*(?:\((\d+)\))?\s*\(([^)]+)\)\s*([<>=!]+)\s*(.+)'
        match = re.search(pattern, line)


        if match:
            state_name = match.group(1)
            derivative_order = int(match.group(2)) if match.group(2) else 1
            condition = match.group(3).strip()
            operator = match.group(4).strip()
            value = match.group(5).strip()


            time_bounds = {}
            if '<=' in condition and '>=' in condition:
                time_pattern = r'([-\d.]+)\s*<=\s*t\s*<=\s*([-\d.]+)'
                time_match = re.search(time_pattern, condition)
                if time_match:
                    time_bounds = {
                        'lower': {'value': float(time_match.group(1)), 'operator': '>='},
                        'upper': {'value': float(time_match.group(2)), 'operator': '<='}
                    }
            elif '>=' in condition:
                time_pattern = r't\s*>=\s*([-\d.]+)'
                time_match = re.search(time_pattern, condition)
                if time_match:
                    time_bounds = {
                        'lower': {'value': float(time_match.group(1)), 'operator': '>='}
                    }


            constraint_value = value
            constraint_type = 'standard'
            if value.startswith('||') and value.endswith('||'):
                constraint_value = value.strip('||')
                constraint_type = 'absolute'


            return {
                'state': state_name,
                'condition': condition,
                'derivative_order': derivative_order,
                'time_bounds': time_bounds,
                'operator': operator,
                'value': constraint_value,
                'type': constraint_type,
            }


        return None


    def _parse_nonlinear_term(self, line, target_dict='nonlinear_terms'):
        """
        Parse nonlinear terms, nonlinear_terms2, or equations.
        target_dict can be: 'nonlinear_terms', 'nonlinear_terms2', or 'equations2'
        """
        if '=' in line:
            parts = line.split('=', 1)
            var_name = parts[0].strip()
            expression = parts[1].strip()
            
            state_pattern = r'state_(\d+)(?!_)'
            action_pattern = r'action_(\d+)(?!_)'
            derivative_pattern = r'state_(\d+)_(\d+)'
            ident_pattern = r'term_(\d+)(?!_)'
            
            state_deps = sorted(set(int(m) for m in re.findall(state_pattern, expression)))
            action_deps = sorted(set(int(m) for m in re.findall(action_pattern, expression)))
            deriv_deps = sorted(set((int(a), int(b)) for a, b in re.findall(derivative_pattern, expression)))
            ident_deps = sorted(set(int(m) for m in re.findall(ident_pattern, expression)))
            
            term_data = {
                'expression': expression,
                'state_dependencies': state_deps,
                'derivative_dependencies': deriv_deps,
                'action_dependencies': action_deps,
                'ident_dependencies': ident_deps,
            }
            
            if target_dict == 'nonlinear_terms':
                self.nonlinear_terms[var_name] = term_data
            elif target_dict == 'nonlinear_terms2':
                self.nonlinear_terms2[var_name] = term_data
            elif target_dict == 'equations2':
                self.equations2[var_name] = term_data


    def get_config(self):
        return self.config


    def get_config_by_section(self, section_key):
        return self.config.get(section_key, None)


    def get_state_constraints(self):
        return self.state_constraints


    def get_ode_requirements(self):
        if not self.derivative_spec:
            return {"states": [], "highest": [], "saved": []}


        states = sorted(self.derivative_spec.keys())
        highest = []
        saved = []


        for state_idx in states:
            orders = self.derivative_spec[state_idx]
            highest.append(max(orders))
            saved.append(orders)


        return {"states": states, "highest": highest, "saved": saved}


    def print_config(self, indent=2):
        print(json.dumps(self.config, indent=indent, default=str))


    def save_config_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"Configuration saved to {filename}")


    def get_calculate_dervative_config(self):
        if not self.derivative_spec:
            return {
                "states": [],
                "highest": [],
                "saved": []
            }


        states = sorted(self.derivative_spec.keys())
        highest = []
        saved = []


        for state_idx in states:
            orders = self.derivative_spec.get(state_idx, [])
            if orders:
                highest.append(max(orders))
            else:
                highest.append(0)
            saved.append(sorted(orders))


        return {
            "states": states,
            "highest": highest,
            "saved": saved
        }



class DomainConfiguration:

    def __init__(self, config):

        self.config = config

    def configure_batch(self):
        fixed_domain = self.config["DOMAIN"].get("upper_bound_time", "")
        type_problem = self.config["REWARD"]["optimization"].get('type',"")
        if fixed_domain:
            self.config["DOMAIN"]["batch_size"] = 1
        if type_problem:
            if type_problem  == "Max":
                self.config["DOMAIN"]["descending"] = False
            elif type_problem == "Min" :
                self.config["DOMAIN"]["descending"] = True
            else:
                raise ValueError(f"The type of pronlem should be 'Max' or 'Min' ")

    def scaling(self):

        reward = self.config["REWARD"]
        target_values = [cond['target_value'] 
                        for cond in reward.get('End_Point_Condition', []) 
                        if isinstance(cond.get('target_value'), float)]
        goal_oriented = [cond['is_endpoint'] 
                        for cond in reward.get('End_Point_Condition', []) 
                        if isinstance(cond.get('is_endpoint'), bool)]
        
        self.config["DOMAIN"]["target_values"] = target_values
        self.config["DOMAIN"]["goal_oriented"] = goal_oriented
    
    def pipeline(self):
        self.configure_batch()
        self.scaling()
        return self.config
    


def create_config(config_data):
    CONFIG_DIR = Path("./config")
    DEFAULT_FILENAME = "problem_domain.json"
    problem_path = os.getenv("PROBLEM")
    if problem_path and Path(problem_path).exists():
        Path(problem_path).unlink()
        print(f"Deleted old config: {problem_path}")
    CONFIG_DIR.mkdir(exist_ok=True)
    output_filename = Path(problem_path).name if problem_path else DEFAULT_FILENAME
    output_path = CONFIG_DIR / output_filename
    with open(output_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"Config created: {output_path}")
    return output_path